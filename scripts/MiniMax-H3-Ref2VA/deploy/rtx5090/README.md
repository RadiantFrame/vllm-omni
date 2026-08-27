# RTX 5090 机器硬件档案(本目录部署目标机)

> 8×RTX 5090 服务器(host104),`2rtx5090/` 与 `4rtx5090/` 所有部署配置的硬件底座。
> 数据来源:`nvidia-smi` / `nvidia-smi topo -m` / `torch.cuda` 实测查询(2026-08-24)+ 官方规格推算,实测项已标注。

## 一、整机配置

| 组件 | 规格 |
|---|---|
| GPU | **8× NVIDIA GeForce RTX 5090**(消费级 Blackwell,GB202,cc 12.0) |
| 显存/卡 | 31.4 GiB GDDR7(512-bit @ 28 Gbps),无 MIG、无 ECC |
| 驱动 / CUDA | 610.43.02 / CUDA 13.3 |
| 功耗 | 600 W/卡(整机 GPU 峰值 ~4.8 kW;满载实测 ~530–540 W/卡,贴功耗墙) |
| CPU | 2× Intel Xeon Gold 6530(32C/颗)→ 64C128T,4 NUMA 节点 |
| 主机内存 | 503 GiB |

## 二、单卡核心构成(三类核心)

| 核心类型 | 数量 | 职责 | 峰值 |
|---|---:|---|---|
| CUDA cores(FP32 ALU) | 21,760(128/SM,共 170 SM,实测) | elementwise / norm / 通用 | FP32 ~105 TF |
| **Tensor Cores(第 5 代)** | **680(4/SM)** | **GEMM 专用**(BF16/FP8/FP4/INT8) | 见下表 |
| RT cores(第 4 代) | 170(1/SM) | 光追,与 DL 无关 | — |

### 理论算力峰值(每卡;除 FP32 外全部来自 Tensor Core)

| 精度 | 稠密 | 稀疏 |
|---|---:|---:|
| FP32(CUDA core) | ~105 TF | — |
| BF16/FP16 Tensor | ~210 TF | 419 TF |
| **FP8 Tensor(当前部署在用)** | **419 TF** | 838 TF |
| FP4/NVFP4(Blackwell 新增,H3 未用) | 838 TF | 1676 TF |
| 显存带宽 | **1.79 TB/s** | — |
| L2 缓存 | ~96 MB | |

算术强度平衡点:FP8 下 419 TF ÷ 1.79 TB/s ≈ **234 FLOP/Byte**——计算导向卡;GEMM 密集段吃满 tensor core,访存段靠大 L2。

## 三、GPU 拓扑与 NUMA(对部署影响最大,实测)

```
配对(NODE=同 PCIe 交换机;跨对 SYS=跨 CPU socket,慢得多):
  GPU0 ↔ GPU1   NUMA0(CPU 0-15, 64-79)
  GPU2 ↔ GPU3   NUMA1(CPU 16-31, 80-95)
  GPU4 ↔ GPU5   NUMA2(CPU 32-47, 96-111)← NIC 在此节点
  GPU6 ↔ GPU7   NUMA3(CPU 48-63, 112-127)
```

- **无 NVLink**,P2P 仅限上述 PCIe 配对;
- **多路服务的 GPU 配对必须按上述四对**(0,1/2,3/4,5/6,7):TP2 的 all-reduce 落在 NODE 内;
  配错(如 0,4)会跨 socket 走 SYS,通信明显变慢;
- 吞吐方案选 **4×TP2** 而非 TP4/TP8 的根本原因:TP4+ 必跨 socket;4 路独立 TP2 各守 NUMA 岛互不干扰
  (实测 4 路并发效率 ~99%);
- 可选优化:每路 worker 进程 `numactl --cpunodebind=$i --membind=$i` 绑到对应 NUMA,
  让 pinned host RAM 的 H2D 走本地内存控制器(未验证收益)。

## 四、工作负载 → 核心的映射(H3/FL2VA)

| 计算部分 | 跑在哪 |
|---|---|
| DiT qkv/out/mlp GEMM(diffuse 绝大头) | **Tensor Core(FP8 W8A8,CUTLASS/DeepGEMM)** |
| attention(CUDNN_ATTN backend) | Tensor Core(FP16/BF16 accumulate) |
| RMSNorm / SwiGLU / 残差 / patchify | CUDA core + 显存带宽(main #6281/#6283 融合在压这部分) |
| VAE decode(卷积) | Tensor Core + CUDA core |

满载时热点基本在 Tensor Core → 优化优先级成立:**FP8(2× tensor 峰值)> compile(消灭空隙)> cache(跳过计算)**。

## 五、对照参考

- vs **H800**(80G SXM):FP8 稠密 ~1979 TF/卡(≈5090 的 4.7×)、NVLink 900GB/s、HBM 3.35TB/s;
  实测吞吐对照:2h800 4 路 8.85 视频/分 vs 2rtx5090 4 路(R=0.04)3.04 视频/分——差距小于算力比,
  拓扑与并发效率部分追回。
- 消费卡限制:无 NVLink(TP 舒服上限 2)、无 ECC(长跑需留意 bit 翻转)、功耗墙先于算力墙(满载 ~540/600W)。
