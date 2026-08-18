# MiniMax-H3 在 2×RTX 5090 上的部署与多路压测实践(vLLM-Omni)

> 对照 H800 篇(`../../h800/README.md`)的姊妹篇。差异核心:H800(80GB)走 FP8+compile 冲延迟;
> **RTX 5090(32GB)显存放不下 BF16 权重,只能走 DLO+eager,优化重心从"并行/精度"转向
> "分辨率/缓存/多路并发"**。本目录还包含 H800 篇没有的 **N 路 2 卡服务并发压测**。

## 一、背景与约束

- **机器**:8×RTX 5090 32GB(Blackwell 消费卡),主机 RAM **503GB**。每路服务固定用 2 卡。
- **模型/负载**:MiniMax-H3 FL2VA,默认压测形状 **480p(832×480)/ 5s / 50 步**(1317 tokens,seed 0),
  客户端 `../../generate/generate_480p_5s.sh`。
- **三条硬约束**:
  1. **32GB 放不下 BF16 全 resident**:TP2 每卡 ~38.8GB → 必须 **DLO**(`--enable-distributed-layerwise-offload
     --dlo-no-use-allgather`),且 DLO 强制 `--enforce-eager`(流式 hook 破坏 cuda-graph)。
  2. **FP8 与 DLO 互斥**(运行期 weight-stride 冲突)→ 本目录全程 BF16;online FP8 的尝试见 `../4rtx5090/`(OOM)。
  3. **多路并发的门槛是主机 RAM,不是 GPU**:DLO 把整套权重(~135GB)以 pinned 内存钉在主机,
     实测**每服务 ~187GB**(权重 135 + 2×worker 各 ~26GB 进程开销),与 `--dlo-resident-layers` 无关。
- **基础拓扑**(所有脚本共用):TP2 + TE-TP2 + USP1/ring1 + VAE pp2/tile/tiling + DLO resident20 +
  enforce-eager + `CUDNN_ATTN`(Blackwell;FA3 是 Hopper 的,5090 上用 cuDNN)。

## 二、实测结果总览(TL;DR)

480p/5s 稳态(第 2 次请求起;run1 因 warmup 略慢):

| 配置 | step latency | E2E | 相对 | 脚本 | 状态 |
|---|---:|---:|---|---|---|
| 1344×768/8s,**无 Cache-DiT**(起点) | 20.7s | ~1036s | — | `deploy.sh` | ✅ 可跑 |
| 480p/5s + Cache-DiT **R=0.04**(基线) | 1.72s | 86s | **~12×** | `deploy_tier0.sh` | ✅ |
| 基线 + R=**0.20** | **1.45s** | **72.5s** | **−16%** | `deploy_tier1_cache_aggressive.sh` | ✅ **单路最优**(画质未验) |
| 基线 + resident 20→32 | 1.73s | 86.3s | **0%(无效)** | `deploy_tier1_resident32.sh` | ✅ 阴性结果 |
| 基线 + resident 40 | — | — | 预期同上 | `deploy_tier1_resident40.sh` | 未测 |
| 基线 + 步级跳过(`scm medium`) | — | — | 预期 >0.20 档 | `deploy_tier1_cache_skip.sh` | 未测 |

**12× 提速的构成**:降分辨率(token 1935→1317)+ Cache-DiT 跳 block;单独归因未拆分。

### 分阶段分解(tier0,PROFILER 实测,稳态)

| 阶段 | 耗时 | 占比 |
|---|---:|---:|
| encode_prompt(Qwen3-VL) | 1.99s | 2.3% |
| **diffuse(49 步 DiT)** | **79.58s** | **93.4%** |
| decode(VAE) | 3.17s | 3.7% |

→ 优化只值得花在 `diffuse`;encoder/VAE 合计 ~5s,忽略。

### 资源占用实测

| 资源 | 数值 |
|---|---|
| GPU HBM(跑请求,resident20) | ~17.4–18.0 GB/卡(峰值;32GB 有余量) |
| GPU HBM(resident32) | ~25 GB/卡 |
| GPU idle(加载完不动) | ~2 GB/进程 |
| **主机 RAM / 每服务** | **~187 GB**(18.7%+18.5% of 503GB,两 worker 各 ~93GB) |

## 三、多路 2 卡服务并发压测(本目录特有)

脚本:每对 GPU 起一个独立 tier0 服务(端口 8000 起),`PAIRS` 数组控制路数:

| 路数 | 脚本 | host RAM | 结果 |
|---|---|---:|---|
| 1 | `deploy_tier0.sh` | ~187GB | ✅ |
| **2** | `deploy_tier0_2svc.sh` | ~374GB | ✅ **本机上限内的推荐档** |
| 3 | `deploy_tier0_3svc.sh` | ~561GB | ⚠️ >503GB,**预测 OOM,未实测** |
| 4 | `deploy_tier0_4svc.sh` | ~748GB | ❌ **实测 OOM**(见下) |

- **4 路 OOM 实录**(503GB 机器):并发加载时权重加载 5s→87s、Model loading 193–202s(内存压力抖态),
  随后某 rank 被 **SIGKILL(exit -9,kernel OOM killer)**,主进程 `Rank N scheduler is dead` → EOFError 退出。
- **降 resident 救不了内存**:resident 层仍保留 pinned CPU master 副本。
- 压测客户端:`../../generate/generate_480p_5s_n.sh`(`PORTS="8000 8001"` 可覆盖)。

### 并发专属坑:30s 输出超时(已修)

两路同时出图时,VAE decode + **MP4 软编码(CPU)** 被争抢 >30s,撞上引擎硬编码的
`_ASYNC_OUTPUT_TIMEOUT = 30.0`(`diffusion_engine.py`)→ 请求被 abort 返回 500。
**视频实际已生成完**(49 步全跑完;超时 cancel 后结果仍送达,日志 `InvalidStateError: CANCELLED` 可证)。

修复:该常量已改为读 `VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT`(默认仍 30);**多路脚本已 export 120**。
单路服务一般不会触发,可不动。

## 四、实践总结

### ✅ 成功点
1. **降分辨率 + Cache-DiT = 最大单项收益**(20.7 → 1.72 s/step,~12×)。
2. **Cache-DiT 阈值 0.04→0.20**:再 −16%(86→72.5s),运行稳定;画质门(LPIPS/PSNR/音频余弦)未跑,上生产前必验。
3. **DLO + Cache-DiT 兼容**:DLO 后端有 `_prev_hook` 回退处理被跳过的 block;官方无此组合基线,本机 49 步全跑通。
4. **CUDNN_ATTN 在 Blackwell 正常**(解析日志 `Resolved ... 'CUDNN_ATTN'` + 端到端出图)。
5. **多路并发 2 路可行**,吞吐翻倍,资源账清晰(见上表)。

### ❌ 失败 / 无效点(避坑)
1. **调 `--dlo-resident-layers` 不提速**:480p 下 diffuse 是 compute-bound,H2D 早已被计算重叠掩盖
   (resident 20→32 实测零收益,仅多占 7GB/卡)。GPU 那 ~14GB 闲余应留给并发/大 shape,不是单请求延迟。
2. **online FP8 不可用**(与 DLO 互斥);在 4×5090 上无 DLO 直跑 FP8 也 OOM(加载期 BF16→FP8 瞬态 2× 峰值,
   详见 `../4rtx5090/README.md`)。
3. **4 路并发必 OOM**(主机 RAM 748GB > 503GB);3 路预测也不行。
4. **并发出图撞 30s 输出超时**(CPU 编码争抢)——用 `VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT` 拉高。
5. **多路同时加载会拖慢磁盘 I/O**:4 路并发加载比单路慢一个数量级,属预期,不是故障。

### 关键教训
- **32GB 消费卡上 DLO 是 BF16 的唯一内存路径**,代价是 enforce-eager(无 compile/cuda-graph);
  想走 FP8+compile 需 ≥80GB 卡(见 H800 篇)或 offline FP8 checkpoint(未建成)。
- **多路并发的预算表**:每服务 ~187GB pinned host RAM;503GB 机器 → 2 路安全 / 3 路悬 / 4 路不可能。
- **单请求延迟看 compute**:分辨率、cache 阈值是杠杆;resident/GPU 显存不是。
- 派生指标 `denoise_step_latency_ms` 之外,拆阶段用 `PROFILER=1 bash deploy_tier0.sh`。

## 五、脚本索引

| 文件 | 配置 | 实测 |
|---|---|---|
| `deploy.sh` | 2 卡 DLO+BF16,**无 cache**(最早版本,1344×768) | 20.7s/step(起点) |
| `deploy_tier0.sh` | 基线:+ Cache-DiT R=0.04 | **1.72s/86s** |
| `deploy_tier1_resident32.sh` / `_resident40.sh` | resident 20→32/40 单变量 | 32:无效;40:未测 |
| `deploy_tier1_cache_aggressive.sh` | R=0.04→0.20 单变量 | **1.45s/72.5s(最优)** |
| `deploy_tier1_cache_skip.sh` | + `scm_steps_mask_policy:medium`(步级跳过) | 未测 |
| `deploy_tier0_2svc.sh` | **2 路**服务(0,1 / 2,3,端口 8000/8001) | ✅ 可跑 |
| `deploy_tier0_3svc.sh` | 3 路服务(+4,5) | ⚠️ 预测 OOM,未实测 |
| `deploy_tier0_4svc.sh` | 4 路服务(+6,7) | ❌ 实测 OOM(exit -9) |

请求客户端在 `../../generate/`:`generate.sh`(1344×768/8s)、`generate_480p_5s.sh`(基线形状)、
`generate_480p_5s_n.sh`(多端口并发压测)。

## 六、剩余杠杆(按 ROI)

1. **cache_skip 步级跳过**:单变量脚本已备好,预期比 R=0.20 更快,画质风险也更大 → 先过质量门。
2. **offline FP8 checkpoint + DLO**:唯一能让 5090 吃到 FP8 提速的路(无加载瞬态、与 DLO 兼容),
   需先离线量化 DiT(62→~31GB);encoder(51.5GB)不在官方 FP8 范围内。
3. **质量门补课**:R=0.20 档与 tier0 同 seed 对照(LPIPS/PSNR + 音频谱余弦),上生产前必做。
4. **更高并发**:需加主机 RAM(≥600GB 跑 3 路、≥768GB 跑 4 路)或减少每服务 pinned 占用。
