# MiniMax-H3 1×H800 部署（单卡最佳推理性能）

> 单卡 80G 上 H3 的唯一可行拓扑：**模型级 CPU 卸载（互斥驻留）+ BF16 + Cache-DiT + FA3**。
> 实测：480p/832×480/5s 稳态 **52.8s**；FP8 在单卡 80G 上不可用（见 §3）。

## TL;DR
| 项 | 结论 |
|---|---|
| 推荐脚本 | **`deploy.sh`**（`--enable-cpu-offload` + BF16 + Cache-DiT high + FA3） |
| 480p/832×480/5s | 稳态 **≈52.8s**（denoise ~43s 占 81%） |
| 768p/1344×768/8s | 未测（denoise ~4×480p，估计 ~200s+，swap 62G/次） |
| 显存 | 模型就绪 58.9 GiB/卡，余量 ~21G |
| 弃用 | FP8（单卡 80G 加载期 OOM，见 §3） |

---

## 1. 目录脚本
| 脚本 | 状态 | 说明 |
|---|---|---|
| `deploy.sh` | ✅ 推荐 | cpu-offload + BF16 + Cache-DiT(high) + FLASH_ATTN(FA3) + regional compile |

## 2. 配置与原理（为什么这样才放得下）
| Flag | 值 | 作用 |
|---|---|---|
| `--num-gpus` | 1 | 单卡 |
| `--enable-cpu-offload` | 开 | **核心**：编码器(BF16 ~51.5G) 与 DiT(BF16 ~62G) 互斥驻留——同一时刻 GPU 只放当前阶段组件，其余躺主机内存（本机 2TB） |
| `--quantization` | **不用**（BF16） | FP8 单卡不可用（§3） |
| `--cache-backend` + config | cache_dit，R=0.04 | 跨步缓存（多卡实测最大单项） |
| `--diffusion-attention-backend` | FLASH_ATTN | Hopper=FA3，本机验证最优 |
| （不加 enforce-eager） | — | regional compile 与 sequential offload 兼容 |
| `VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT=300` | 环境变量 | offload 下 post-denoise 超 30s 默认值会被误杀 |

**互斥驻留的显存账**：任一时刻峰值 = max(编码阶段 ~52G, 去噪阶段 DiT 62G+激活, VAE 阶段) 而非求和 ~120G——这就是单卡能跑的原因。

## 3. 为什么单卡不能用 FP8（实测根因）
| 项 | 事实 |
|---|---|
| 现象 | `--quantization fp8` + cpu-offload 启动即 OOM（`logs/h3_0818_1h800.log` 首跑，已覆盖） |
| 机制 | 在线 FP8 的 BF16→FP8 转换（`scaled_fp8_quant`，CUDA kernel）在**加载阶段逐层跑在 GPU 上**；逐层“量化后回 CPU”未能阻止累积 → 编码器 51.5G 常驻 + DiT 转换产物累积到 **78.05G** → 差 250MB 爆掉 |
| 为何 pro6000 的 FP8 版能跑 | 那是 **96G** 卡，扛住了转换累积窗口 |
| 其它路也不通 | FP8+DLO 文档声明不兼容（权重 stride） |
| 代价 | BF16：每次 swap 搬 62G（FP8 是 31G）+ 无 FP8 GEMM；**质量反而是参考级最好** |

## 4. 实测汇总（2026-08-18，`logs/h3_0818_1h800.log`，480p/832×480/5s/50 步）
| # | e2e | 说明 |
|---|---:|---|
| 1 | 109.0s | compile warmup（不计） |
| 2 | 85.8s | 懒初始化未收敛（不计） |
| 3–9 | **52.73–53.08s** | 稳态（7 次，波动 ±0.3s） |

- denoise ~43s（**占 E2E ~81%**）；其余 ~10s = 编码 + 2 次 swap(62G over PCIe) + VAE + MP4。
- 显存：加载 10.3 GiB → 模型就绪 58.91 GiB（余量 ~21G）。
- **跨卡（同 480p workload）：2×H800 TP2=26.5s → 1×H800=52.8s = 1.99×，近完美线性减半。**
- 生效锚点：`Resolved ... 'FLASH_ATTN'` ✓ / `Cache-dit enabled successfully` ✓ / 0 报错。

## 5. 使用
| 环境变量 | 默认 | 说明 |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | 0 | 单卡选择 |
| `PORT` | 9000 | |
| `MODEL` | `.../MiniMax-H3/FL2VA` | Ref2VA 换 `.../Ref2VA` 重启（勿同跑） |

```bash
bash deploy.sh
bash ../../generate/generate_480p_5s.sh   # warmup 2 + 计时 ≥3 取末次（本配置收敛需 2 个请求）
```

## 6. 调参入口
| 优先级 | 杠杆 | 预期 / 风险 |
|---|---|---|
| ① | Cache-DiT 阈值 0.04→0.10/0.20 | denoise 占 81%，收益上限可观；**必过质量门** |
| ② | 768p 压测 | 显存余量 21G 够，但 swap/激活随分辨率涨，先小后大 |
| ✗ | FP8 | 单卡 80G 不可用（§3），勿再试 |

## 7. 注意事项
| 事项 | 说明 |
|---|---|
| 收敛需 2 个请求 | req1=compile、req2=懒初始化，稳态从 req3 起 |
| swap 是 I/O 型开销 | 每请求 2 次 62G PCIe 搬运；若其它进程挤占 PCIe 带宽会直接抬高 E2E |
| 与 2/4 卡不可直比 | 各配置 workload/精度不同；跨卡对比只在同 workload 下有效（480p 链：26.5 vs 52.8s） |
| 单卡 offload 不吃 TP/USP | 那些切分对单卡无意义；VAE patch-parallel 保持 1 |
