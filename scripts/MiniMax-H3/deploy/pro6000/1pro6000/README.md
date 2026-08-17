# MiniMax-H3 在 1×RTX PRO 6000 上的部署实践（vLLM-Omni）

> 姊妹篇：`../../h800/4h800/README.md`（80GB 数据卡走 FP8+compile 冲延迟）、
> `../../rtx5090/2rtx5090/README.md`（32GB 消费卡走 DLO+并发）。
> 本篇是 **单卡 96GB + 小主机内存（125GB）** 的第三种形态：模型级 offload + FP8。

## 一、机器与内存账（一切设计的出发点）

- **机器**：1× RTX PRO 6000 Blackwell（96 GiB, SM120），主机 RAM **125 GiB**，swap 63 GiB。
- **模型**：MiniMax-H3 FL2VA 分区（135G on disk）= DiT 62G + Qwen3-VL encoder ~48-63G + VAE 10.4G。
- **三条硬约束**：
  1. BF16 全 resident 124G > 96G 显存 → 官方 PRO 6000 recipe 明确 "TP1 is not an option"；
  2. DLO 需要把权重钉在主存（≥200G RAM）→ 125G 主机不可行；
  3. 模型级 offload 的换向瞬间两个组件副本共存：BF16 下 48+62=110G，贴着 125G 红线 →
     实测多次 OOM/冻结。

## 二、最终方案（deploy_fp8.sh，实测 4 连发全绿）

**模型级 CPU offload + 在线 FP8 + Cache-DiT + tmux + 停用 systemd-oomd + cgroup 保险丝**：

| 开关 | 作用 |
|---|---|
| `--num-gpus 1 --enable-cpu-offload` | 组件互斥换向（DiT↔encoder 不同时驻留 GPU） |
| `--quantization fp8` | **关键**：DiT 62G→31G，换向峰值 110G→~91G，另赚 Blackwell FP8 GEMM 提速 |
| `--cache-backend cache_dit`（R=0.04） | 跨步缓存，H3 "high" 档 |
| `--diffusion-attention-backend CUDNN_ATTN` | SM120 无 TRTLLM_ATTN；PRO 6000 recipe 钉定值 |
| `VLLM_OMNI_PIN_CPU_MEMORY=0` | CPU 副本不锁页（可 swap）——**必需**：稳态 worker ~104G 若全部 pin 死，125G 主机无回收余量 |
| `VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT=300` | 去噪后的 VAE 解码/编码期 > 默认 30s 会被误杀 |
| tmux 运行 | 服务脱离 VS Code 进程树（桌面崩溃不连带） |
| `systemd-oomd` 停用 | **本机必做**，见下 |
| `systemd-run -p MemoryMax=105G` | cgroup 保险丝：失控只杀服务，不冻结整机（稳态已 ~101G） |

### FP8 与 offload 的兼容性说明

FP8 文档只声明与 **layerwise（DLO）** offload 互斥（weight stride 问题）；模型级
（sequential）offload 无此限制，且 `--enable-cpu-offload` 使加载走 `load_device="cpu"`
路径（日志 `Online quantization with CPU offload, using cuda for weight loading` +
`Selected CutlassFP8ScaledMMLinearKernel`），无 GPU 端量化瞬态。本组合官方未验证，
本机实测：加载 82s、4 连发 200 OK、画质门沿用官方 FP8 结论
（LPIPS 0.116 / PSNR 23.6 / 音频余弦 0.96；上生产前建议自跑质量门）。

## 三、实测结果（480p 832×480 / 5s / 50 步）

| 配置 | E2E（稳态） | denoise | 状态 |
|---|---:|---:|---|
| BF16 裸奔（无内存治理） | 178.8s | 3.65s/step | 第 2 次请求 OOM，VS Code 被 kill |
| BF16 + unpinned（+两段式 pageout 补丁，已移除） | 207.8s | 4.16s/step | 换向峰值 109-110G，贴 125G 红线，不可靠 |
| **FP8（deploy_fp8.sh）** | **86-93s** | **1.6-1.9s/step** | ✅ **4 连发全绿** |

## 四、踩坑全记录（按时间序，避免重蹈）

1. **30s 异步输出超时**：去噪完成后 VAE 解码+MP4 编码超过引擎默认
   `_ASYNC_OUTPUT_TIMEOUT=30s`，视频已生成但响应被 abort（InvalidStateError:
   CANCELLED）。→ `VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT=300`。
2. **pinned 内存不可回收**：offload 副本默认 pin_memory，换向双副本尖峰无法被内核
   换出 → OOM killer 杀 VS Code。→ 引擎补丁 `VLLM_OMNI_PIN_CPU_MEMORY=0`。
3. **换向双副本 110G 尖峰**：BF16 下曾实现两段式 madvise(MADV_PAGEOUT) 补丁试图
   压峰值，但一次内核 direct-reclaim 换页风暴仍把整机拖入僵死（硬重启），补丁已
   移除。→ 正解是 FP8 把峰值砍到 ~91G。
4. **systemd-oomd 是隐形团灭凶手**：Ubuntu 默认开启，按 **PSI 压力指标**（非内存耗尽）
   杀整个 cgroup scope——任何依赖 swap 的方案都会触发它。`systemctl disable` 只管
   开机自启，**必须 `sudo systemctl stop systemd-oomd.socket systemd-oomd.service`**
   （必要时 `mask`）。判别方法：journal 出现 `systemd-oomd killed N process(es)`。
5. **服务跑在 VS Code 终端 = 连坐**：VS Code 被 kill 时其 scope 内 42 个进程（含
   vllm）一起死。→ tmux。
6. **cgroup 保险丝阈值要留余量**：MemoryMax=108G 在合法峰值 109G 上误杀（差 1.2G），
   105G 的 FP8 版注意 cgroup 记账含页缓存，过紧会先于真实 OOM 触发。
7. **端口 8000 被 VS Code 端口转发占用**：用 9000，客户端 `BASE_URL` 覆盖。
8. **worker 启动期偶发 SIGILL（exit -4）**：2026-08-17 出现两次（均在刚杀掉长跑
   服务后），内核日志 `trap invalid opcode in python3.12`；之后带/不带额外环境
   变量均正常启动。未定论（疑与重度服务退出后的初始化状态相关）——**遇到先重试
   一次**；脚本已带 `PYTHONFAULTHANDLER=1`，复发时日志会有 Python 级崩溃栈。

## 五、关于 mmap+DLO（未采用的架构级方案）

单卡理论最优（权重以 mmap 视图直读页缓存，host RSS≈0），但当前代码**双重不可用**：
- mmap 路径仅在 `DLO+AllGather 且 dp>1` 时激活（单卡必然 no-allgather+dp=1），
  代码留有 "rank-local mmap path" TODO；
- H3 显式 `_supports_mmap_loading=False`（grouped-QKV 重排 / fused-MLP 打包在
  mmap 路径未验证等价性）；
- 且 `_shard_and_pin` 最终仍复制私有 CPU shard（dp=1 时=全量），mmap 只省加载期 RSS。

若未来要支持 128G 级小主机，实现该 TODO（rank-local mmap 流式 + H3 布局适配器 +
权重等价性验证）是正解，适合作为独立 PR。

## 六、剩余杠杆（按 ROI）

1. **Cache-DiT 阈值 0.04→0.10/0.20**：5090 实测 −16%，本拓扑直接可套；必过质量门
   （LPIPS/PSNR + 音频谱余弦，固定 prompt/seed）。
2. **升分辨率**：FP8 后 GPU 去噪期仅 ~45G 占用，720p（1248×704）有余量，E2E 预计
   按像素比例线性放大。
3. **FP8 质量门自测**：与 BF16 同 seed 对照（当前引用官方数据）。
4. **Ref2VA**：换 `MODEL=.../MiniMax-H3/Ref2VA` 重启，两分区勿同起。

## 七、脚本索引

| 文件 | 配置 | 实测 |
|---|---|---|
| `deploy_fp8.sh` | FP8 + 模型级 offload + Cache-DiT（**主推**） | ✅ 4 连发 86-93s |
| `deploy.sh` | BF16 回退/对照（unpinned + 保险丝） | ⚠️ 峰值 109-110G 贴红线，仅作对照 |
| 客户端 | `../../generate/generate_480p_5s.sh`（`BASE_URL=http://localhost:9000`） | |

引擎侧配套补丁（本仓库）：
- `VLLM_OMNI_PIN_CPU_MEMORY`（offloader/base.py，默认关闭不影响上游行为）
