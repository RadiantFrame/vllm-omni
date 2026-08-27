# MiniMax-H3 在 4×RTX 5090 上的单服务部署实践(vLLM-Omni)

> H800 篇(`../../h800/README.md`)与 2×5090 篇(`../2rtx5090/README.md`)的姊妹篇。
> 差异核心:本目录是 **4 卡起一个 TP4 服务**(不是 2rtx5090 目录的 N 路 2 卡多服务);
> 且仓库内**没有任何验证过的 4×5090 拓扑**(官方 recipe 只验证 1/2 卡,H800 脚本基于 80GB 卡),
> 本篇记录从零推导拓扑 → 实测验证 → 踩坑的完整过程。

## 一、背景与内存账

- **机器**:8×RTX 5090 32GB(Blackwell),主机 RAM 503GB;本配置固定用其中 **4 卡**(`0,1,2,3`)。
- **模型构成**(BF16,磁盘实测):DiT/transformer 62G + Qwen3-VL encoder 51.5G(retained 50 层)
  + video/audio VAE 10.4G(VAE 权重在 patch-parallel 下**每卡复制**)。
- **每卡权重账(TP4)**:

| 组件 | 全量 | TP4 每卡 |
|---|---:|---:|
| DiT | 62G | 15.5G |
| encoder(TE-TP4) | 51.5G | 12.9G |
| VAE(replicated) | 10.4G | 10.4G |
| **合计** | | **~38.8G > 31.4G 上限** |

→ **BF16 无 offload 必 OOM**,这是全部设计的出发点。

### 三条硬约束
1. **BF16 必须走 DLO**(no-AllGather 在任意 TP 下被接受,且是 `--dlo-resident-layers` 的前置);
   DLO 强制 `--enforce-eager`(流式 hook 破坏 cuda-graph 捕获)。
2. **online FP8 与 DLO 互斥** → 想用 FP8 就必须去掉 DLO,而去掉 DLO 后全 resident 38.8G 装不下(见实测)。
3. **拓扑只有两种合法形态**:TP4+USP1(DiT group 占满)或 TP2+USP2;`--usp 4 --tp 1` 是反模式
   (每卡复制完整 DiT,recipe 明确警告);TE-TP 合法值为 1/2/4(64/8 头整除约束)。

## 二、拓扑设计(候选矩阵)

| 候选 | 组合 | 每卡权重 | 判定 |
|---|---|---:|---|
| **A(采用)** | TP4 + DLO no-AG + BF16 | 38.8G→DLO offload | ✅ **`deploy.sh`,唯一确定可行** |
| B | TP2 + USP2 + DLO | 同为 DLO(权重按 TP2 切) | 备选;activation 换切分,480p 无必要 |
| C | TP4 + online FP8,**无 DLO** + regional compile | 量化后 ~31G(全 resident) | ❌ **实测 OOM**(`deploy_fp8.sh`) |
| D | `--usp 4 --tp 1` | 全 DiT ×4 复制 | ❌ 反模式,直接排除 |
| (参照) | PRO 6000 96GB:TP2 无任何 offload | 77.5G/卡 | 换大卡才是"去 DLO+开 compile"的正解 |

采用配置(`deploy.sh`):`--num-gpus 4 --tensor-parallel-size 4 --text-encoder-tp-size 4 --usp 1 --ring 1
--vae-patch-parallel-size 4 --vae-parallel-mode tile --vae-use-tiling + DLO(resident 20)+ enforce-eager +
Cache-DiT(R=0.04)+ CUDNN_ATTN + num-weight-load-threads 8`。
TE-TP4 是关键:encoder 默认全压 rank0,TE-TP4 把它 4 路切分,显著拉平 rank0 峰值。

## 三、实测结果

| 脚本 | 结果 | 关键数据 |
|---|---|---|
| `deploy.sh`(TP4+DLO+BF16) | ✅ **可跑** | 峰值 HBM **~10G/卡**(GPU 大量闲余,见下);去噪延迟未记录,建议用 `generate/generate.sh` 实测(预期优于 2 卡的 1.72s/step) |
| `deploy_fp8.sh`(TP4+online FP8,无 DLO) | ❌ **加载期 OOM** | 见下方详细 |

### FP8 失败实录(错误签名 + 机制)

- **报错**:`torch.OutOfMemoryError ... 30.53 GiB allocated`,位置
  `fp8.py:159 process_weights_after_loading → ops.scaled_fp8_quant(layer.weight)`,4 个 worker 全部同点崩。
- **机制**:online FP8 是**加载时量化**——BF16 权重先上 GPU,再逐层转 FP8,转换瞬间 **BF16 原件 + FP8 输出同时驻留**;
  且无 DLO ⇒ encoder/VAE 的 on-demand staging 不触发(offload backend 为 None)⇒ 全模型试图 resident(38.8G/卡)> 31.4G → OOM。
- **重要澄清**:FP8 kernel 本身在 5090(Blackwell)**没问题**——日志确认
  `Selected CutlassFP8ScaledMMLinearKernel` + `DeepGEMM E8M0 enabled`。失败纯粹是**容量**,不是兼容性,调参无解。
- 该脚本保留给 ≥80GB 卡用;5090 上别再试。

### 显存利用:~10G/卡 峰值说明什么

TP4 + DLO 下每 resident block 仅 ~1/4 块大小(20 块 resident ≈ 6G + buffer/context ≈ 10G),
**每卡 ~22G 闲余**。但(承接 2rtx5090 目录的实测结论)480p 是 compute-bound,
**调高 resident 换不来单请求延迟**(2 卡上 resident 20→32 实测零收益)。闲余显存的正确用途:
**并发/吞吐**或**更大 shape**(1344×768/更长时长),不是单请求提速。
极限是 resident=50(全 DiT resident ≈ 15.5G/卡,encode/decode 期峰值 ~28–30G,理论可装),如要试需盯峰值。

### 主机内存(估算,未实测)

单服务 4 worker:pinned 权重 ~135G(4 rank 分摊)+ 4×~26G 进程开销 ≈ **~240G**,
503G 机器单服务无压力(2rtx5090 目录的实测值是每服务 187G,TP2 双 worker,可类比)。

## 四、实践总结

### ✅ 成功点
1. **TP4+DLO+BF16 在 4×32GB 上确定可行**——从内存账推出的唯一解,实测可跑。
2. **TE-TP4 拉平 encoder 峰值**(51.5G 的最大显存块 4 路切分)。
3. **CUDNN_ATTN / Cache-DiT / VAE pp4 tile / DLO 的 `_prev_hook` cache 回退** 全部正常(延续 2rtx5090 验证)。

### ❌ 失败 / 无效点(避坑)
1. **online FP8 在 32GB 卡不可行**(上述容量账;≥80GB 卡才考虑,且那时应参照 H800 篇的候选 A)。
2. **`--usp 4 --tp 1` 反模式**:每卡复制全 DiT,显存直接爆炸,不要因为"USP 切序列"而误用。
3. **别为单请求延迟调 resident**(compute-bound;闲显存留给并发/大 shape)。
4. **显存峰值低 ≠ 有问题**:~10G/卡 是 TP4+DLO 的正常形态,不是配置错误。

### 关键教训
- 32GB 卡的设计空间被"BF16 全 resident 38.8G/卡"这一条锁死:要么 DLO(损失 compile),
  要么 FP8(损失 DLO,且加载期瞬态更吃显存),**二选一没有免费午餐**;两个都要只能换 ≥80GB 卡。
- 无验证基线时,先算**每卡权重账**(全量 ÷ TP + VAE replicated)再选拓扑,能直接排除一半候选。
- 失败要记**精确错误签名**(`fp8.py:159 scaled_fp8_quant` + 分配量),下次一眼对上。

## 五、脚本索引

| 文件 | 配置 | 状态 |
|---|---|---|
| `deploy.sh` | TP4 + DLO(resident20) + BF16 + Cache-DiT R=0.04 + enforce-eager | ✅ 主推 |
| `deploy_fp8.sh` | TP4 + online FP8 + 无 DLO + regional compile(**勿在 5090 用**) | ❌ OOM 存档 |

请求客户端在 `../../generate/`(`generate.sh` 默认打 8000)。
两脚本均支持 `PROFILER=1` 开关(`--enable-diffusion-pipeline-profiler`)。

## 六、剩余杠杆(按 ROI)

1. **补去噪延迟实测**:同 prompt/seed 跑 `generate.sh`,与 2 卡 1.72s/step 对照,量化 TP4 收益。
2. **Cache-DiT R=0.04→0.20**:2 卡实测 −16%,本拓扑直接可套(先过质量门)。
3. **offline FP8 checkpoint + DLO**:5090 上吃 FP8 提速的唯一路径(无加载瞬态、与 DLO 兼容),
   需先离线量化 DiT(62→~31G);encoder 不在官方 FP8 范围。未建成。
4. **并发/大 shape**:每卡 ~22G 闲余是现成的吞吐空间(多请求并发或回 1344×768)。
