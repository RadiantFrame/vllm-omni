# MiniMax-H3 在 H800 上的推理加速实践（vLLM-Omni）

> 本文整合自同目录此前的 `OPTIMIZATION_PLAN.md`（分层优化方案）与 `ACCELERATION_SUMMARY.md`（实践总结），
> 是 8 卡 H800 机器上 MiniMax-H3（FL2VA）加速实践的完整记录：**方案 → 脚本 → 实测 → 结论**。

## 一、背景与目标

- **机器**：8×H800 80GB（Hopper SM90）。实践固定使用其中 **4 卡**（`CUDA_VISIBLE_DEVICES=4,5,6,7`）。
- **模型/负载**：MiniMax-H3 **FL2VA**（首帧→视频+音频，中长序列 ~124–209 帧 / 1344×768），**单请求端到端延迟最小**为首要目标。
- **允许**：FP8 量化 + Cache-DiT/TeaCache 跨步缓存（在质量门限内）。
- **为什么需要专门方案**：H3 是 CFG-distilled 联合视频+音频扩散 DiT（~33B DiT + ~25.75B Qwen3-VL 编码器，BF16 各约 66/51.5 GiB）。官方 4-GPU recipe（USP4+VPP4+compile+FLASH_ATTN）在 **B300(144GB)** 上标定，rank-0 峰值 ~117–130GB，**80GB 的 H800 放不下**，必须重新组合「权重切分 + 序列切分 + 精度」。

### 三条硬约束
1. **H800 = Hopper SM90**：支持 FP8（CUTLASS W8A8）、FA2/FA3；**不支持** FA4 / cuDNN / TRTLLM / SageAttn / NVFP4 / quack / MX（Blackwell 或 NPU 专属）。
2. **H3 是 CFG-distilled**：每步仅 1 次 forward、无负样本分支 → `--cfg-parallel-size` 硬性为 1，**CFG-Parallel 不可用**。
3. **80GB 显存**：USP 只切序列不切权重 → 纯 USP4 BF16 必 OOM；必须 **FP8**（DiT 权重减半）或 **HSDP/TP**（切权重）才 fit。

### H3 自带能力矩阵（`docs/user_guide/diffusion_features.md`）
| TeaCache | Cache-DiT | SP(Ulysses+Ring) | CFG-Parallel | TP | PP | HSDP | Layerwise Offload | VAE-Patch-Parallel | Quant | Step-Exec |
|---|---|---|---|---|---|---|---|---|---|---|
| ✓(FL2VA) | ✓ | ✓ | **✗(CFG-distilled)** | ✓(DiT/TE) | ✗ | ✓ | ✓ | ✓(tile) | ✓(DiT) | ✗ |

排除项：CFG-Parallel、Pipeline-Parallel、MagCache/StepCache（未验证）、所有 Blackwell/NPU 专属后端。

## 二、实测结果总览（TL;DR）

**从 BF16 基线 174.47s 压到 114.77s（−34%）。** 延迟阶梯（FL2VA，50 步，稳态取第 3 次请求）：

| 配置 | 稳态 E2E | 相对增量 | 对应脚本 |
|---|---:|---|---|
| 候选 D：USP4+HSDP4+TE-TP4，**BF16** | 174.47s | 基线（官方 Hopper 基线） | `deploy_tier3_D.sh`（≡`tier0/tier1`） |
| 候选 A：USP4+TE-TP4，**FP8** | 155.74s | **−10.7%**（FP8 GEMM + 免 HSDP all-gather） | `deploy_tier3_A.sh`（≡`tier2`） |
| 候选 A + **Cache-DiT**（DBCache, R=0.04） | **114.77s** | **−26%（vs A）/ −34%（vs D）** ← **当前最优** | `deploy_tier4_4b.sh` |

### 2 路并发（`deploy_tier4_4b_2svc.sh`，8 卡每路 4 卡，480p/832×480/5s）
单请求稳态 **≈16.15s**（全场最低延迟），聚合 7.49 视频/分；三布局终局对比见 §四之「8 卡机三布局对比」。

### 当前最优配置（114.77s）
```bash
vllm serve .../MiniMax-H3/FL2VA --omni --trust-remote-code --num-gpus 4 \
  --usp 4 --ring 1 \
  --text-encoder-tp-size 4 \
  --quantization fp8 \
  --cache-backend cache_dit \
  --cache-config '{"Fn_compute_blocks":1,"Bn_compute_blocks":0,"max_warmup_steps":4,"residual_diff_threshold":0.04,"max_continuous_cached_steps":1,"enable_taylorseer":false}' \
  --enable-cache-dit-summary \
  --diffusion-compile-granularity regional \
  --vae-patch-parallel-size 4 --vae-parallel-mode tile --vae-use-tiling \
  --diffusion-attention-backend FLASH_ATTN        # 不加 --enforce-eager → regional compile
```

## 三、目录内容（脚本索引）

| 文件 | 配置 | 实测 |
|---|---|---|
| `deploy.sh` | 候选 C + FP8（TP2+USP2+**TE-TP4**） | 可跑（sglang tp2+ulysses2 对应物） |
| `deploy_tier0.sh` | 候选 D 基线（BF16, HSDP4） | 174.47s |
| `deploy_tier1.sh` | tier0 + Tier1 旋钮显式钉住 | ≈tier0（无延迟差，属预期） |
| `deploy_tier2.sh` | 候选 A（FP8） | 155.74s |
| `deploy_tier3_{A,B,C,D}.sh` | 并行候选矩阵 | A=155.74 / D=174.47；B、C 未测 |
| `deploy_tier4_4a.sh` | A + TeaCache(0.17) | 未测 |
| `deploy_tier4_4b.sh` | A + Cache-DiT(high) | **114.77s（最优）** |
| `deploy_tier4_4b_2svc.sh` | 2 路并发（8 卡，每路 4 卡 = tier4_4b 配置，端口 9000–9001，master 29500+i） | 480p：**16.15s/路（延迟全场最低）** |
| `deploy_tier5_attn_flash.sh` | 最优 + 本地 FA3 | 114.77s（参考） |
| `deploy_tier5_attn_flashhub.sh` | 最优 + HF FA3（`FLASH_ATTN_3_HUB`） | 117.64s（略慢） |
| `deploy_tier5_attn_flashinfer.sh` | 最优 + FlashInfer | **崩**（H3 不兼容，仅存档） |
| `deploy_tier5_kvcache_fp8_{skip45,skip23,noskip}.sh` | KV-cache FP8 三档 | **no-op**（cuda 不支持，仅存档） |
| `../../generate/generate.sh` | FL2VA 客户端（固定 prompt/seed，打 8000） | 计时用：warmup 1 + 计时 ≥3 |

## 四、实践总结

### 2 路并发实测（`deploy_tier4_4b_2svc.sh`，2026-08-19，480p/832×480/5s）

压测口径：`NUM_SERVICES=2 PORT_BASE=9000 bash ../../generate/generate_480p_5s_nsvc.sh`（2 路同时发同一请求，共 7 轮；日志 `logs/deploy_tier4_4b_2svc.svc<N>.gpu<..>.log`，0 报错）。

**每轮 × 2 路 E2E（秒）：**

| 轮次 | svc0 (gpu0-3) | svc1 (gpu4-7) | 均值 | 最大值 |
|---|---:|---:|---:|---:|
| 1（compile warmup） | 33.90 | 33.78 | 33.84 | **33.90** |
| 2 | 15.82 | 16.05 | 15.94 | **16.05** |
| 3 ⚠️ | 16.98 | 17.10 | 17.04 | **17.10** |
| 4 | 15.92 | 15.91 | 15.91 | **15.92** |
| 5 | 15.98 | 15.97 | 15.97 | **15.98** |
| 6 | 15.92 | 16.08 | 16.00 | **16.08** |
| 7 | 16.02 | 16.01 | 16.01 | **16.02** |
| **稳态均值（2–7 轮）** | 16.44 | 16.52 | **16.15** | — |

- 剔除第 3 轮同步抖动（两路同时 +1.1s 后即恢复，外部瞬态）后稳态 ≈ **15.98s**；两路差异仅 ±0.04s，无梯度。
- 2 路并发几乎无衰减（GPU 常驻权重 + USP4，对照单路 480p 外推基线 ~15.7–15.8s，衰减 ≈ 1–2%）。

**8 卡机三布局终局对比（同 480p/5s workload，全部实测）：**

| 布局 | 单请求稳态 | 聚合吞吐 | tail（最慢路） | 定位 |
|---|---:|---:|---:|---|
| 8×1 卡（`../1h800/deploy_8svc.sh`，offload） | 61.5s | 7.61 视频/分 | ~64s | 仅「每路单卡」硬约束 |
| **4×2 卡（`../2h800/deploy_4svc.sh`，TP2）** | 27.25s | **8.85 视频/分** 🏆 | 28.8s | **吞吐最优** |
| **2×4 卡（本脚本，USP4）** | **16.15s** 🏆 | 7.49 视频/分 | 17.1s | **延迟最优** |

> 吞吐口径：`路数 ÷ 最后一轮最大延迟 × 60`（同步轮次下整轮墙钟由最慢路决定，故用末轮 max 而非均值；末轮 max：8×1=63.08s、4×2=27.13s、2×4=16.02s）。

- **延迟 vs 吞吐的明确取舍**：2×4 卡把单请求压到 16s（是 4×2 卡的 1.69× 快），但吞吐反而低 15%（7.49 vs 8.85/分）——路数减半而每路加速不到 2×（TP2 26.5s → USP4 16s = **1.66×**，USP4 的 all-to-all + 固定开销吃掉线性）。选型只看「要最快出片（2×4）还是单位时间出片最多（4×2）」。
- 隐含跨卡数斜率（480p 实测）：TP2 26.5s → USP4 16s，2× 卡换 1.66× 加速——「加卡值不值」的实测依据。

### ✅ 成功点（带来实测收益）
1. **FP8 优于 BF16（A vs D，−10.7%）**：FP8 GEMM 在 Hopper 更快，且 A 不带 HSDP 的逐层 all-gather。在线 FP8 确认生效（日志 `Selected CutlassFP8ScaledMMLinearKernel for Fp8PerTensorOnlineLinearMethod`）。
2. **Cache-DiT DBCache 是最大单项收益（155.74→114.77s，~1.35×）**：`residual_diff_threshold=0.04` 保守档；靠 `--enable-cache-dit-summary` 确认「确在 skip 步」。
3. **regional torch.compile**：默认开启（不加 `--enforce-eager`），稳态收益计入；首请求 ~30s 编译作 warmup 排除。
4. **VAE pp4 + tile、FLASH_ATTN(FA3)**：基座正常，`diffuse` 主路径最优。
5. **自动生效（无 flag）**：fused RMSNorm+RoPE（#5801）、视频帧转换内存上界（#5732）。

### ❌ 失败 / 无效点（避坑清单）
1. **TE-TP 中间值（4 卡上设 2）→ 崩**：`GroupCoordinator` 的 `cpu_group` 断言对非组内 rank 失败。代码限制：`text_encoder_tp_size` 只能取 **1 或 world_size(4)** → 候选 C 改 TE-TP4 才跑通。
2. **FlashInfer（`FLASHINFER_ATTN`）→ H3 不可用**：H3 的 packed 多模态自定义 mask 令 `segment_packbits` 算出负维 → 首请求崩。
3. **diffusion KV-cache FP8 → H800 上 no-op**：`FLASH_ATTN` 在 cuda 不支持 fp8 KV（`_supported_kv_cache_dtypes={"npu":{"fp8"}}`，仅 NPU）→ 静默禁用，三档脚本等同基线。
4. **`FLASH_ATTN_HUB` 名字混淆**：它是 **FA2**（拉 `kernels-community/flash-attn2`），非 FA3；且需 `pip install kernels` 否则静默回退。真 HF FA3 是 **`FLASH_ATTN_3_HUB`**。
5. **HF FA3 → 略慢**：117.64s vs 本地 FA3 114.77s（+2.5%），还多 354MB 下载与 `kernels` 依赖，不换。
6. **FP8 + HSDP 叠加**：H3 未验证（`hsdp_fp8.py` 补丁未为 H3 验证）→ 不混用。
7. **对推理延迟 ≈ 0 的项**：`--num-weight-load-threads`（仅启动）、`--diffusion-compile-granularity regional`（等于默认）、异步 diffusion 输出（吞吐向）、`--enable-cache-dit-summary`（纯日志）。

## 五、完整优化方案（Tier 0–6，方法论存档）

### Tier 0 — 冻结测量基线（难度 ★）
- 固化三条命令：server 启动（Tier 3 候选）、单请求（`../../generate/generate.sh`，固定 prompt/seed/shape/steps）、重复基准（warmup 1 + 计时 ≥3）。
- **首请求 = regional compile warmup，不计入稳态**；正式测延迟时关 profiler。

### Tier 1 — 零成本 / 纯 flag 运行期优化（难度 ★）
| # | 优化 | 做法 | 实测注 |
|---|---|---|---|
| 1.1 | regional torch.compile | **不加** `--enforce-eager`（默认 regional；`full` 与 SP/HSDP/offload 不兼容） | ✅ 生效 |
| 1.2 | VAE patch-parallel 4 + tiling | `--vae-patch-parallel-size 4 --vae-parallel-mode tile --vae-use-tiling`（H3 VAE 仅支持 `tile`） | ✅ 生效 |
| 1.3 | FLASH_ATTN 后端 | `--diffusion-attention-backend FLASH_ATTN`（Hopper 解析为 FA3，经 `fa3-fwd` 包） | ✅ 最优 |
| 1.4 | 多线程权重加载 | `--num-weight-load-threads N` | ⚠️ 仅启动加速 |
| 1.5 | 异步 diffusion 输出 | 保持 `--step-execution` 关闭（默认） | 吞吐向 |
- 自动生效：fused RMSNorm+RoPE（#5801）、视频帧转换内存上界（#5732）。

### Tier 2 — FP8（难度 ★★，flag + 质量门）
- `--quantization fp8`（在线 W8A8：磁盘 BF16 不变，加载时量化，dynamic 激活）。
- 官方验证：峰值 −22%（68.52→53.51 GiB），LPIPS 0.116 / PSNR 23.6 / 音频余弦 0.96 全过门。
- 量化范围：DiT 的 qkv/out/mlp/condition/adaln linears；patch/timestep/final 投影保 FP32；**编码器(BF16)与 VAE(FP32) 不动**。
- 敏感层保 BF16：`--diffusion-quantization-config '{"transformer":{"method":"fp8","ignored_layers":[...]}}'`。
- ⚠️ 与 layerwise offload(DLO) 互斥；与 HSDP 未验证不混用。
- **实测**：A(FP8) 比 D(BF16+HSDP4) 快 10.7% —— 在本机上 FP8 是延迟优选。

### Tier 3 — 并行策略（难度 ★★★）
80GB 必须「权重切分 + 序列切分」：

| 候选 | 组合 | 显存/卡 | 实测 |
|---|---|---:|---|
| **D**（官方 Hopper 基线，BF16） | `--usp 4 --ring 1 --use-hsdp --hsdp-shard-size 4 --text-encoder-tp-size 4` | ~54–55GB | **174.47s**（质量最高档） |
| **A**（FP8） | `--usp 4 --text-encoder-tp-size 4 --quantization fp8` | ~55–56GB | **155.74s（延迟胜出）** |
| B（显存极小） | `--tensor-parallel-size 4 --text-encoder-tp-size 4 --quantization fp8` | ~30–45GB | 未测；TP 全序列 attention + 50 层 all-reduce，偏慢 |
| C（均衡，sglang 镜像） | `--tensor-parallel-size 2 --usp 2 --text-encoder-tp-size 4 --quantization fp8` | ~45–55GB | 未测（TE-TP4 形式可跑） |

- **实测结论：延迟优先选 A；质量优先选 D。** A/B 规则：一次一个变量，固定 prompt/seed/shape/steps。
- VAE patch-parallel 必须 =1 或 = DiT group size（4 卡即 4）。
- **modular pipeline（可选）**：`vllm serve <repo根>` + `--task-type fl2va`，一个服务跑 T2VA/FL2VA/Ref2VA（请求 `extra_params.task` 路由）；注意 TeaCache 仅 FL2VA。

### Tier 4 — 跨步缓存（难度 ★★★★，调参 + 质量门）
**TeaCache 与 Cache-DiT 二选一**（一个服务一个 cache backend）。
- **4a TeaCache**（仅 FL2VA，`#5840`）：`--cache-backend tea_cache --cache-config '{"rel_l1_thresh":0.17}'`（0.17 为 H3 默认；阈值越大越快质量越掉）。官方未公布 H3 TeaCache 质量数，需自测门。**本次未实测。**
- **4b Cache-DiT**（本次采用）：H3「high」档 = `Fn=1,Bn=0,W=4,R=0.04,MC=1`（H200 官方实测 1.35×/SSIM 0.9709/PSNR 34.98）。**本机实测 155.74→114.77s。**
  - 调参：要更快升 `residual_diff_threshold` 到 0.10–0.30（必重过质量门）；更激进加 `"scm_steps_mask_policy":"medium"`（50 步→约 33 步实算）。
  - 请求级 `quality=high|lossless`（#5853）：同一服务混跑加速/参考质量请求（标准延迟部署不需要）。
- **质量门（强制）**：同 prompt/seed/shape 对比有/无 cache，LPIPS/PSNR + 音频谱余弦 + RMS（H3 含音频务必查）；`--enable-cache-dit-summary` 看每步命中/跳过。TaylorSeer 保持关闭（不适合蒸馏模型）。

### Tier 5 — 注意力后端 / KV-cache（难度 ★★★★）
- **注意力后端 A/B（已收官）**：本地 `FLASH_ATTN`=**FA3**（`fa3-fwd`，唯一装了的 FA 包）✅ 最优；`FLASH_ATTN_HUB`=HF **FA2**（回退方向）；`FLASH_ATTN_3_HUB`=HF **FA3**（117.64s，略慢）；`FLASHINFER_ATTN`（H3 自定义 mask 崩）。**结论：注意力层已到顶，用本地 FLASH_ATTN。**
- **diffusion KV-cache FP8（H800 死方向）**：`--diffusion-kv-cache-dtype fp8` 在 cuda+FLASH_ATTN 下被静默禁用（仅 NPU 支持）；FlashInfer 虽支持但 H3 不可用；TRTLLM 的 fp8_e4m3 是 Blackwell 专属。

### Tier 6 — 显存余量策略（难度 ★★★★，按需）
- **DLO（resident=8）**：候选 D 基础上加 `--enable-distributed-layerwise-offload --dlo-no-use-allgather --dlo-resident-layers 8 --enforce-eager` → 峰值 ~36GB 且延迟几乎不变（官方 H100 实测 38.2 vs 38.3s）。**与 FP8 互斥**；兼容矩阵见 `docs/design/feature/distributed_layerwise_offload.md`。
- **HSDP**：已作为候选 D 的切分手段；独立 HSDP 不切序列，长序列不如 D。

### 候选汇总（含实测）
| Tier | 优化 | 预期 | 实测 |
|---|---|---|---|
| 0 | 冻结基线 + warmup | 度量前提 | ✅ |
| 1 | compile + VAE pp4 + FA3（+自动融合） | 中大 | ✅（计入基线） |
| 2 | online FP8 | 显存 −22%、延迟↓ | ✅ **−10.7%** |
| 3 | 并行 D/A/B/C | 大（diffuse） | ✅ **A 胜 D**（155.74 vs 174.47） |
| 4 | TeaCache / Cache-DiT | 1.35–2.5× | ✅ **Cache-DiT −26%（114.77s）** |
| 5 | 注意力后端 / KV FP8 | 中 | ❌ 后端已到顶；KV FP8 不可用 |
| 6 | DLO / 独立 HSDP | 显存余量 | 未需（55GB 有余量） |

## 六、剩余杠杆与未来方向（按 ROI）
1. **Cache-DiT 阈值调高**（0.04→0.10/0.20）：**最大未拿项**，多 skip 步进一步降延迟；必带质量门。
2. **TeaCache(0.17) 对照**：FL2VA 专属，与 Cache-DiT 二选一，未实测，值得 A/B。
3. **H3 迁移 mask-free varlen 路径**（未来方向）：`#4645` 已为 Flux2/HunyuanVideo1.5 实现「skip attention-mask → 免 varlen 自定义 mask」，H3 尚未跟进；若迁移可省二次 mask 物化开销，并可能解锁 FlashInfer 等后端。
4. **4-step 蒸馏 sigma schedule（#5991）**：NPU 专属；概念上 50→4 步是数量级提升，若移植 CUDA 则是 game-changer。

## 七、关键教训
- **USP 不切权重**：80GB 必须 FP8 或 HSDP/TP 才放得下；纯 USP4 BF16 必 OOM。
- **FA3 已是 Hopper 注意力上限**：别在注意力层再花时间（HF FA3/FlashInfer 都不会更快，后者还崩）。
- **`nvidia-smi` 高占用 ≠ 权重**：模型常驻 ~54GiB + compile/激活 + 分配器预留到 `gpu_memory_utilization≈0.9`；FP8 已把 DiT 权重减半（否则 BF16 复制 4 份早 OOM）。
- **默认日志的 `denoise_step_latency_ms` 是派生值**（stage_gen ÷ 步数）；拆 `text_encoder/diffuse/vae.decode` 用 `--enable-diffusion-pipeline-profiler`。
- **首请求 = compile warmup**，计时一律排除。
- **精度分布要分清**：DiT+编码器 = BF16（DiT 可选 FP8），VAE = FP32，patch/timestep/final/输出头 = FP32。

## 附录 A：关键文件 / 代码位置
- 模型：`vllm_omni/diffusion/models/minimax_h3/{minimax_h3_transformer,pipeline_minimax_h3,encoder,vae,quality_policy}.py`
- 编译：`vllm_omni/diffusion/compile.py`（regionally_compile）、`worker/diffusion_model_runner.py`
- Cache-DiT：`vllm_omni/diffusion/cache/cachedit/{backend,config,model_specific,runtime}.py`；TeaCache：`cache/teacache/{config,extractors,backend}.py`
- 注意力：`attention/backends/{flash_attn,flash_attn_hub,flashinfer_attn}.py`（KV 量化支持判断在 `attention/layer.py`）、`platforms/cuda/platform.py`（backend 解析）
- 并行：`diffusion/distributed/{parallel_state,sp_plan,hsdp}.py`；量化：`quantization/factory.py`、`diffusion/quantization/hsdp_fp8.py`
- **官方 4×H100 perf 基线**：`tests/dfx/perf/tests/test_minimax_h3_vllm_omni.json`
- recipe：`recipes/MiniMaxAI/MiniMax-H3.md`；特性矩阵：`docs/user_guide/diffusion_features.md`；FP8：`docs/user_guide/quantization/fp8.md`

## 附录 B：测量与验证方法
1. 每档：起服务 → `../../generate/generate.sh` warmup 1 次（不计）→ 计时 ≥3 取末次稳态 E2E。
2. 看分阶段：`PROFILER=1 bash deploy_*.sh`（`--enable-diffusion-pipeline-profiler`）。
3. 质量门（FP8/缓存/量化相关必跑）：固定 seed 对照，LPIPS/PSNR + 音频谱余弦（门限参照 `recipes/MiniMaxAI/MiniMax-H3.md` 的 validated 段）。
4. 确认开关生效的日志锚点：FP8 → `Selected CutlassFP8ScaledMMLinearKernel`；Cache-DiT → `Cache-dit enabled successfully` + summary；注意力 → `Resolved diffusion attention backend '<名字>'`（警惕 `Falling back` 字样）。
