# MiniMax-H3 2×H800 部署（最佳推理性能）

> 推导自 4×H800 实践赢家（[`../4h800/`](../4h800/) `deploy.sh`，114.77s@768p/8s）。
> 实测已验证：**2 卡最优并行是 TP2（不是 4 卡的 USP）**。

## TL;DR
| 事项 | 结论 |
|---|---|
| 推荐脚本 | **`deploy.sh`**（TP2 + FP8 + Cache-DiT） |
| 480p/832×480/5s | 稳态 **≈26.5s** |
| 768p/1344×768/8s | 稳态 **≈216.6s**（TP2 唯一能跑；USP2 OOM）。4 卡为 114.77s → **1.89×，近线性** |
| 模型常驻/卡 | 50.4 GiB（80GB 余量充足） |
| 弃用 | `deploy_usp2.sh`（480p 慢 9.8%、768p OOM，仅存档） |

---

## 1. 目录脚本
| 脚本 | 状态 | 说明 |
|---|---|---|
| `deploy.sh` | ✅ 推荐 | TP2 + TE-TP2 + FP8 + Cache-DiT(high) + VAE pp2 + FA3 + regional compile |

## 2. 推荐配置（deploy.sh）
| Flag | 值 | 作用 |
|---|---|---|
| `--tensor-parallel-size` | 2 | DiT 权重+头 2 路切分（~16.5GB/卡 FP8） |
| `--text-encoder-tp-size` | 2 | 编码器 2 切（= world_size，合法；中间值会撞 bug） |
| `--quantization` | fp8 | 在线 W8A8：DiT 权重减半 + FP8 GEMM 提速 |
| `--cache-backend` + `--cache-config` | cache_dit，`R=0.04` high 档 | 跨步缓存（4 卡实测 −26%，本档最大收益项） |
| `--vae-patch-parallel-size` / `--vae-parallel-mode` | 2 / tile | VAE 并行解码（须 = DiT group size；H3 仅支持 tile） |
| `--diffusion-attention-backend` | FLASH_ATTN | Hopper 上解析为 FA3（fa3-fwd），已验证最优 |
| `--diffusion-compile-granularity` | regional（默认，不加 enforce-eager） | regional torch.compile；首请求 = warmup |
| `--cfg-parallel-size` | 1（默认） | H3 是 CFG-distilled，硬约束 |

## 3. 并行选型：为什么 TP2（而非 4 卡赢家 USP）
| 维度 | TP2 ✅ | USP2 ❌ |
|---|---:|---:|
| DiT 权重 | **切半**（~16.5GB/卡） | 全量复制（~33GB/卡） |
| 模型常驻/卡 | **50.4 GiB** | 65.8 GiB |
| 每卡算力 | 头切半 | 序列切半（2 卡规模下两者等效） |
| 通信 | 每层 all-reduce | all-to-all（固定开销） |
| 480p 稳态 | **26.5s** | 29.1s（慢 9.8%） |
| 768p | ✅ 可跑 | ❌ OOM |

**依据**：USP 不切权重，2 卡上显存代价致命；480p 短序列下序列切分的二次方收益小，抵不过 all-to-all 固定开销 → **与 4 卡结论（USP4 胜）相反，属卡数相关**。TP2+TE-TP2 亦为仓库已验证的 2-GPU 拓扑。

## 4. 实测汇总
| 配置 | workload | 结果 | 显存/卡 | 日志 |
|---|---|---|---|---|
| TP2（deploy.sh） | 480p/832×480/5s/50步 | ✅ 稳态 **26.5s**（26.35/26.58/26.51；warmup 40.5s），denoise ~0.53s/it | 模型 50.4 / 进程 51.2 GiB | `logs/h3_0817_2h800.log` |
| TP2（deploy.sh） | 768p/1344×768/8s/50步 | ✅ 稳态 **216.6s**（216.03/217.30/216.53；warmup 229.4s），denoise ~4.33s/it；对比 4 卡 114.77s = 1.89×（近线性） | 模型 50.4 / 进程 51.2 GiB（与 480p 一致） | `logs/h3_0818_2h800_768p_tp.log` |

**生效锚点（TP2 run 验证）**：`Selected CutlassFP8ScaledMMLinearKernel` ✓ / `Cache-dit enabled successfully` ✓ / `Resolved diffusion attention backend 'FLASH_ATTN'` ✓。

## 6. 使用
| 环境变量 | 默认 | 说明 |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | 4,5 | 8 卡机选 2 张 |
| `PORT` | 9000 | 服务端口 |
| `NUM_WEIGHT_LOAD_THREADS` | 8 | 启动加速，不影响推理 |
| `PROFILER` | 0 | =1 开分阶段计时（text_encoder/diffuse/vae.decode） |

```bash
bash deploy.sh                        # 起服务
bash ../../generate/generate_480p_5s_nsvc.sh   # 压测：warmup 1 + 计时 ≥3 取末次

bash deploy_4svc.sh                   # 4 路吞吐版（8 卡，每路 2 卡，端口 9000–9003）
NUM_SERVICES=4 PORT_BASE=9000 bash ../../generate/generate_480p_5s_nsvc.sh   # 4 路并发压测
kill $(cat logs/deploy_4svc.pids)     # 4 路全停
```

## 7. 调参入口（按 ROI）
| 优先级 | 杠杆 | 做法 | 预期 / 风险 |
|---|---|---|---|
| ① | Cache-DiT 阈值 | `residual_diff_threshold` 0.04→0.10/0.20 | 再压延迟；**必过质量门**（固定 seed 对照 LPIPS/PSNR+音频余弦） |
| ② | TeaCache 替换 | `--cache-backend tea_cache --cache-config '{"rel_l1_thresh":0.17}'` | FL2VA 专属，与 Cache-DiT 二选一，值得 A/B |
| ✗ | 不可用项 | FlashInfer（H3 自定义 mask 崩）/ KV-cache FP8（cuda+FLASH_ATTN 静默禁用）/ FP8+HSDP（未验证） | 继承 4h800 结论 |

## 8. 注意事项
| 事项 | 说明 |
|---|---|
| 跨 workload 不可比 | 4h800 的 114.77s 是 768p/8s，2h800 的 26.5s 是 480p/5s；只在各自 workload 内比相对值 |
| 首请求 = compile warmup | 计时一律排除 |
| `image_pixels` 字段 | ≠ 输出画布（480p 实测 309,504 ≠ 832×480=399,360），分辨率以请求参数/ffprobe 为准 |
| nvidia-smi ~73GB | 可能只是分配器 0.9 预留（=81.5×0.9）；真实需求看日志 `Process-scoped GPU memory` |
