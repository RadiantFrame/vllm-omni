# MiniMax-H3 在 2×RTX 5090 上的部署(vLLM-Omni)

> 8×RTX 5090(32 GiB/卡,主机 RAM 503 GiB)机器,每路服务固定 2 卡。
> 压测形状:FL2VA **480p(832×480)/ 5s / 50 步**;Cache-DiT 用官方推荐档 **R=0.04**。
> 客户端 `../../generate/generate_480p_fanout.sh`(`NUM_SERVICES` / `PORT_BASE` 控制)。

## 一、服务配置(`deploy.sh`)

```bash
--num-gpus 2 --tensor-parallel-size 2 --text-encoder-tp-size 2
--usp 1 --ring 1 --vae-patch-parallel-size 2 --vae-parallel-mode tile --vae-use-tiling
--quantization fp8                          # 全局:DiT + Qwen3-VL encoder 都量化(online W8A8)
--enable-cpu-offload                        # model-level:阶段互斥换入换出
--diffusion-compile-granularity regional    # cuda graph(无 --enforce-eager)
--cache-backend cache_dit --cache-config '{...R=0.04...}'   # 官方 "high" 档
--diffusion-attention-backend CUDNN_ATTN    # Blackwell
--num-weight-load-threads 8
```

**设计逻辑**:
- FP8 全 resident 仍超 32G/卡(DiT 15.5 + enc 12.9 + VAE 10.4 ≈ 38.8G),offload 必须;
  但 FP8 后**整个 DiT 能常驻**(15.5G/卡),故用 **model-level** 而非逐块流式:
  denoise 49 步零逐块 H2D,encoder/VAE 仅在阶段边界换入换出(每请求 ~0.3–0.6s)。
- model-level 的 swap hook 只在阶段边界触发,DiT 内部 per-block 编译不受影响 → regional compile 可用。
- 配置生效的日志锚点:`Selected CutlassFP8ScaledMMLinearKernel` /
  `Enabling offloader backend: ModelLevelOffloadBackend` /
  `Regional compilation applied to 52 module(s)` / `Cache-dit enabled successfully` /
  cache 上下文 `DBCache_F1B0_W4I1M0MC1_R0.04`。

## 二、实测结果(2026-08-20,R=0.04)

### 单路(`deploy.sh`,日志 `logs/deploy_R0.04_0820_2rtx5090.log`,7 轮)

| 轮次 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| e2e | 94.7s | 85.4s | **77.9s** | 78.0s | 78.0s | 78.0s | 78.0s |

- **稳态 ≈ 78.0 s/请求**(第 3 轮起;round1 含 compile warmup,不计)。
- 单路吞吐 ≈ **0.77 视频/分**。

### 4 路并发(`deploy_4svc.sh`,端口 9000–9003,日志 `logs/deploy_4svc.svc{0..3}.*.log`,7 轮)

每路 e2e:round1 ~95s(并发加载+warmup)→ round2 ~87s → **round3 起稳态 ~78.8s**。

**末轮(第 7 轮)**:svc0 79.06 / svc1 78.87 / svc2 78.86 / svc3 78.41 s
→ **均值 78.80 s,最大 79.06 s**(4 路离散极小)。

**聚合吞吐**(口径:路数 ÷ 末轮最大延迟 × 60):

$$4 ÷ 79.06 × 60 = \textbf{3.04 视频/分}$$

- 并发效率:4 路单路 78.8s vs 单服务 78.0s,衰减仅 ~1%,效率 ≈99%
  ——各路独占 GPU 对,CPU 侧争抢(MP4 编码)由 `VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT=120` 兜住。

## 三、资源占用

| 资源 | 数值 |
|---|---|
| GPU HBM | 各阶段互斥(model-level 换入换出);denoise 期 FP8 DiT 常驻 15.5G/卡 + activation |
| 主机 RAM / 服务 | **~120 GiB**(FP8 权重 ~67G pinned + 2×worker 开销 ~52G) |
| 4 路合计 | **~480G vs 503G,贴边可行**(实测 7 轮无 OOM;启动前 `free -g` 确认) |

## 四、踩坑与注意

1. **encoder FP8 无验证基线**:DiT FP8 有官方质量门,encoder FP8 是新能力——
   上生产前同 seed 对照(LPIPS/PSNR + 音频谱余弦)。
2. **首请求 = compile warmup**(94.7s vs 稳态 78s),测延迟从第 3 轮起算。
3. **4 路是贴边配置**:加载期并发读盘 + 量化瞬时内存更高;
   若 worker exit code -9(kernel OOM killer)即降路数。
4. **多路启动必须钉死 `MASTER_PORT`**(29500+i,`deploy_4svc.sh` 已内置):
   否则 torch.distributed TCP store 随机端口,并发启动撞 EADDRINUSE。
5. **多路并发必须 `VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT=120`**(已内置):
   并发 MP4 软编码超默认 30s,请求实际已完成却 HTTP 500
   (错误签名:diffusion_engine `TimeoutError` + result pump `InvalidStateError: CANCELLED`)。
6. 若换模型/版本后图捕获崩(首请求报 cuda graph 错误),先加回 `--enforce-eager` 定位。

## 五、脚本索引

| 文件 | 状态 | 说明 |
|---|---|---|
| `deploy.sh` | ✅ **推荐(单路)** | 上述配置;稳态 78.0s |
| `deploy_4svc.sh` | ✅ **推荐(吞吐)** | 4 路并发(端口 9000–9003,master 29500+i);3.04 视频/分 |

压测:`NUM_SERVICES=N PORT_BASE=9000 bash ../../generate/generate_480p_fanout.sh`
(单路 `NUM_SERVICES=1`)。
