# LE SSERAFIM Ref2VA Benchmark — 参考资源准备包

面向 **MiniMax H3 Ref2VA** 的 LE SSERAFIM-specific MV generation benchmark。
本包用于下载并整理生成一条"全新、非原 MV 复刻"的 15 秒 K-pop 女团 MV 所需的
**6 张图片 + 3 个视频 + 3 个音频**参考资源，并生成按 Ref2VA 顺序命名的文件清单。

## H3 Ref2VA 输入约束（本包已对齐）

| 约束 | H3 要求 | 本包 |
|---|---|---|
| Images | ≤ 9 | 6 |
| Videos | ≤ 3 clips，每个 2–15s，同类合计 ≤ 15s | 3 × 5s = **15s** |
| Audio | ≤ 3 clips，每个 2–15s，同类合计 ≤ 15s | 3 × 5s = **15s** |
| 文件总数 | ≤ 12 | 6+3+3 = **12** |

脚本自动将完整视频/音频**裁剪为 5s 片段**（可用 `--clip-secs` 在 2–5 间调整，
超过 5 会自动收缩以保证 3 段合计 ≤ 15s）。完整素材保留在 `raw/` 供复现/重截。

## 目录结构

```
le-sserafim-ref2va-benchmark/
├── prepare_references.sh      # 一键下载/准备/裁剪脚本
├── download_bili_api.py       # B 站纯 API 下载器（view+WBI playurl，绕开网页风控）
├── README.md                  # 本文件
├── generation_prompt.txt      # 提交给 H3 的生成 prompt（含分镜与约束）
└── references/
    ├── raw/       完整素材（v1/v2/v3 完整 mp4 + live 完整音轨）
    ├── images/    img1_sakura.jpg ... img6_group.jpg   （6 张）
    ├── videos/    v1_antifragile_choreo.mp4 ... v3_easy_cinema.mp4  （各 5s，提交用）
    ├── audio/     audio1_vocal.m4a ... audio3_live_ambience.m4a     （各 5s，提交用）
    └── manifest/  references_manifest.json
```

## 参考映射（与 Ref2VA 输入槽一一对应）

| 输入槽 | 内容 | 角色 | 默认源（已核验） |
|---|---|---|---|
| Image 1 | Sakura（宮脇咲良） | identity：脸型、发型、身形、外貌 | 官方/高清图 |
| Image 2 | Kim Chaewon（金采源） | identity | 官方/高清图 |
| Image 3 | Huh Yunjin（许允真） | identity | 官方/高清图 |
| Image 4 | Kazuha（中村一叶） | identity | 官方/高清图 |
| Image 5 | Hong Eunchae（洪恩採） | identity | 官方/高清图 |
| Image 6 | 五人团体 concept 照 | 队形 / 造型 / 视觉概念 | 官方/高清图 |
| Video 1 | ANTIFRAGILE Choreography ver.（B站 BV1mP411A7Qo） | 编舞、走位、队形、动态 | B站 API → 5s 片段 |
| Video 2 | UNFORGIVEN 官方MV 4K（B站 BV1dPPXzNEKk） | 舞台表现、表情、表演能量 | B站 API → 5s 片段 |
| Video 3 | EASY 官方MV（B站 BV1tx4y1k7Bu） | 运镜、构图、镜头运动 | B站 API → 5s 片段 |
| Audio 1 | V1 同窗口音轨 | 人声：唱法、乐句、情绪 | 与 Video1 同步截取 |
| Audio 2 | V2 同窗口音轨 | 音乐：BPM、编曲、K-pop 质感 | 与 Video2 同步截取 |
| Audio 3 | Coachella 完整现场（B站 BV1Pr421574f）30%处 | 氛围：观众、舞台、环境声 | B站 API 音轨 → 5s 片段 |

## 快速开始

```bash
# 安装依赖（Debian/Ubuntu 示例）
apt-get install -y python3 ffmpeg curl

# 全量：下载 raw 完整素材 → 裁剪 5s 片段 → 生成 manifest
./prepare_references.sh

# 仅图片 / 仅视频音频 / 预览
./prepare_references.sh --images-only
./prepare_references.sh --videos-only
./prepare_references.sh --dry-run

# 覆盖 / 指定目录 / 1080p / 片段长度 / 统一起始偏移 / 无损音频
./prepare_references.sh --force --out ./refs --max-h 1080 \
    --clip-secs 4 --start-offset 30 --audio wav
```

## 脚本行为说明

- **下载统一走 B 站纯 API**（`download_bili_api.py`）：只调 B 站公开 API
  （view → WBI 签名 playurl → 直链下载 → ffmpeg 合并），**不抓取 B 站视频网页**，
  对云服务器/数据中心 IP 被 HTTP 412 风控的环境完全免疫（已实测通过）。
  内置 4 个经 API 核验的官方 bvid，运行时无需搜索。
- **裁剪**：完整视频/音轨进 `raw/`；提交片段默认**取素材 30% 处 5s**
  （`calc_start = duration * 0.3`，避开片头 logo 与片尾 credits；对 8 分钟长 MV
  如 UNFORGIVEN，中段 50% 会落入片尾 credits，30% 更安全）。
  - `--start-offset S`：全局统一从第 S 秒截取。
  - 每视频独立配置：脚本顶部 `V1_START` / `V2_START` / `V3_START`（默认 -1=自动），
    可精确指定每个视频的有效段落起始点（如副歌/舞蹈段）。
  - Audio1/Audio2 与 Video1/Video2 使用**同一时间窗口**，保证音画同步。
  - 视频片段输出 h264 + yuv420p + AAC（H3 最兼容），音频 m4a/AAC 320k。
  - 旧布局迁移：若 `videos/` 下存在 >15s 的旧完整版，自动移到 `raw/`，避免重复下载。
- **音频**：Audio1/2 从 V1/V2 完整视频音轨截取；Audio3 用 API 的 audio-only 模式
  直接拉取 Coachella 现场音轨（避免下载 43 分钟整段视频），再取 30% 处 5s。
- **图片**：默认指向已人工核验身份的官方/高清素材（Image1–6 均经视觉确认）；
  支持替换为任意 URL 或本地绝对路径。
- **幂等**：已存在且校验通过的文件自动跳过；`--force` 强制重下/重截。
- **校验**：图片校验 MIME+尺寸，视频校验时长，裁剪片段校验时长 ≤ 15s。
- **产物**：`references/manifest/references_manifest.json` 汇总所有槽位文件名、
  来源 URL、H3 约束与生成 prompt。

## 已验证项（全部实测通过）

- ✅ 脚本语法（bash -n）与全部参数解析
- ✅ 6 张图片端到端下载、逐张身份核验（Sakura/Chaewon/Yunjin/Kazuha/Eunchae/团体）、尺寸校验
- ✅ 3 个完整视频 + live 完整音轨在 412 风控环境下端到端下载成功（存入 raw/）
- ✅ **裁剪出的 3 视频 + 3 音频片段均为 5s，合计各 15s，满足 H3 约束**
  （视频 h264/yuv420p，音频 AAC）
- ✅ Audio1/Audio2 与 Video1/Video2 同一时间窗口截取，音画同步
- ✅ 4 个 B 站 bvid 经 view API 逐一核验（标题/时长/UP 主均匹配官方 MV/编舞/现场）

## 使用与合规提示

- 素材均为 LE SSERAFIM / HYBE（Source Music）官方或公开转载内容，仅用于
  研究/基准测试目的，请勿再分发或商用，遵守 B 站平台条款与版权。
- 生成时请严格遵循 prompt 中
  "Create a new original MV sequence. Do not reproduce or copy any existing
  LE SSERAFIM music video shot-for-shot." 的要求，避免输出对既有 MV 的逐镜复刻。
- 建议将 6 张图片替换为**同一概念期**的官方成套概念照，可提升五人身份与
  团体造型的一致性（当前默认图来自不同回归期，仅作可运行占位）。
