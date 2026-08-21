# MiniMax-H3 在 1×RTX PRO 6000 上的部署（vLLM-Omni，FP8 全驻留）

> 姊妹篇：`../../h800/4h800/README.md`（80GB 数据卡）、`../../rtx5090/2rtx5090/README.md`（32GB 消费卡）。
> 本篇是**单卡 96GB** 形态：在线 FP8（DiT + text encoder）+ 全驻留，无需 CPU offload。

## 一、机器与配置

- **机器**：1× RTX PRO 6000 Blackwell（96 GiB, SM120），主机 RAM 125 GiB。
- **模型**：MiniMax-H3 FL2VA 分区（135G on disk）= DiT 62G + Qwen3-VL encoder ~48G + VAE 10.4G。

BF16 全驻留需要 ~124G > 96G 显存，本不可行；**encoder 加入在线 FP8 后**
（DiT 62→31G + encoder 48→~24G + VAE），驻留显存降到 ~69G，96G 单卡直接放下，
CPU offload 及其全部主机内存治理（pin/swap/cgroup 保险丝）不再需要。

## 二、最终方案（deploy_fp8.sh）

| 开关 | 作用 |
|---|---|
| `--quantization fp8` | **关键**：DiT + encoder 在线量化，驻留 124G→~69G，另赚 Blackwell FP8 GEMM 提速 |
| `--num-gpus 1`（无 offload） | 全组件常驻 GPU，无换向、无主机内存压力 |
| `--cache-backend cache_dit`（R=0.04） | 跨步缓存，H3 "high" 档 |
| `--diffusion-attention-backend CUDNN_ATTN` | SM120 无 TRTLLM_ATTN；PRO 6000 recipe 钉定值 |
| `--diffusion-compile-granularity regional` | 区域级 torch.compile |
| `VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT=300` | 去噪后的 VAE 解码/编码期 > 默认 30s 会被误杀 |
| tmux 运行 | 服务脱离 VS Code 进程树 |
| 端口 9000 | 8000 被 VS Code 端口转发占用 |

加载：在线量化 + 权重载入共 ~69s，驻留显存 **68.9 GiB / 96 GiB**（余量可上 720p）。

## 三、实测结果（2026-08-21，480p 832×480 / 5s / 50 步）

客户端 `generate_480p_5s_nsvc.sh`（seed=0），13 轮全部 200 OK。

**以第 3 轮为准：E2E 89.7s，denoise 1.79s/step。**

其后各轮稳定在 88.3-90.5s（denoise 1.76-1.81s/step），说明第 3 轮已是稳态。

### 性能上限说明：功率墙

该卡推理期 100% 利用率下功耗顶死 600W 上限（SW Power Cap Active），
SM 频率被压到 **2235 MHz**（标称 max 3090），温度 88°C。步延迟由功耗预算下的
实际频率决定，运行间波动 ±5% 属正常；连续多轮取稳态值，勿以单轮低点作为基线。

## 四、剩余杠杆（按 ROI）

1. **FA4 替换 CUDNN_ATTN**：Blackwell 原生 kernel（sm_120 预计比 cuDNN 快 ~20%），
   FP16 无精度损失；用 `benchmarks/diffusion/bench_attention_backends.py` 实测选型。
2. **Cache-DiT 阈值 0.04→0.10/0.20**：5090 实测 −16%；必过质量门
   （LPIPS/PSNR + 音频谱余弦，固定 prompt/seed）。
3. **升分辨率**：驻留 69G/96G，720p（1248×704）有余量，E2E 预计按像素比例线性放大。
4. **FP8 质量门自测**：与 BF16 同 seed 对照（当前引用官方数据：
   LPIPS 0.116 / PSNR 23.6 / 音频余弦 0.96）。
5. **Ref2VA**：换 `MODEL=.../MiniMax-H3/Ref2VA` 重启，两分区勿同起。
6. **散热**：88°C 偏高，改善风道/环境温度可提高功率墙内频率。

## 五、脚本索引

| 文件 | 配置 | 实测 |
|---|---|---|
| `deploy.sh` | FP8 全驻留 + Cache-DiT + CUDNN_ATTN（**主推**） | ✅ 稳态 E2E ~89.7s（480p/5s） |
| 客户端 | `../../generate/generate_480p_5s_nsvc.sh`（`BASE_URL=http://localhost:9000`） | |
