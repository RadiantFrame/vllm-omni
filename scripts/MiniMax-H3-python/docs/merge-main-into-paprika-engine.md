# 将 main 合并进 paprika-engine 的标准流程

> 适用场景:paprika-engine 分支吸收上游 origin/main 的更新。
> 核心原则:**直接在 paprika-engine 上 merge,tag 作安全网,不开牺牲性的 sync/* 临时分支**;merge message 手写并点名 paprika-engine,让历史自描述。
> 记录日期:2026-11-16

## 远端布局

| 远端 | 地址 | 权限 |
| --- | --- | --- |
| `origin` | `git@github.com:vllm-project/vllm-omni.git` | 只读(push 已 DISABLE) |
| `rf` | `git@github.com:RadiantFrame/vllm-omni.git` | 可推(fork) |

## 第 0 步 · 前置检查

```bash
git status --short                 # 工作区必须干净
git branch --show-current          # 确认在 paprika-engine
git fetch origin                   # 拿最新 main
git fetch origin --tags            # 拿远端 tag(不带则 setuptools_scm 版本号锚点过期,
                                    #  editable 安装后 __version__ 停留在旧 tag 的 devN,
                                    #  并触发 vLLM-Omni vs vLLM 版本不匹配警告)
git rev-list --left-right --count paprika-engine...origin/main   # 看 ahead/behind
```

有未提交改动时先 commit 或 stash,不要带脏工作区进 merge。

> merge 后如果 `python -c "import vllm_omni; print(vllm_omni.__version__)"`
> 报出的版本仍停留在旧 tag(如 `0.29.0rc2.devN`),说明 tag 没跟上:
> 补 `git fetch origin --tags` 后重新 `VIRTUAL_ENV=$PWD/.venv uv pip install -e .`
> 重新生成 `_version.py`。该差异只影响显示与告警,不影响功能。

## 第 1 步 · 打安全网 tag

```bash
git tag paprika-pre-sync-$(date +%Y%m%d)
```

这是唯一的回滚锚点:

- merge **中途**失败 / 想放弃 → `git merge --abort`(回到 merge 前,tag 都用不上);
- merge **提交后**后悔 → `git reset --hard paprika-pre-sync-YYYYMMDD`。

## 第 2 步 · 直接在 paprika-engine 上 merge

```bash
git merge origin/main
```

要点:

- 冲突就地一次解完,不跨会话留半成品 merge 状态。
- 想先预览冲突再决定,可用 `git merge --no-commit --no-ff origin/main`,
  然后 `git diff --name-only --diff-filter=U` 看冲突清单,不满意 `git merge --abort`。
- 冲突高频区:本分支的 `scripts/TurboH3-python/`、`scripts/MiniMax-H3*/` 改动
  与 main 侧同目录 / 同一模型(`vllm_omni/diffusion/models/minimax_h3/`)的更新。

## 第 3 步 · 手写 merge commit message

不要接受 git 生成的默认信息,按下面模板手写,**必须点名 paprika-engine**:

```
Merge origin/main into paprika-engine

Absorb upstream <old-main>..<new-main>: <一句话概括 main 侧主要内容>。

Conflict resolution:
- <逐条列出冲突文件、以哪边为准、为什么>

Co-Authored-By: Claude Code <noreply@anthropic.com>
```

> 为什么不开临时分支:merge commit 会永久记录"当时所在的分支名",
> 在 `sync/*` 临时分支上做的 merge 会把过期分支名永远留在 paprika-engine 的历史里。

## 第 4 步 · 验证

```bash
# 1) import 冒烟
python -c "import vllm_omni"

# 2) 冲突相关目录的功能冒烟(按本分支改动选)
cd scripts/TurboH3-python && python <常用 preset/export 命令> --help

# 3) 状态确认
git log --oneline -3                                        # merge commit 在顶
git rev-list --left-right --count paprika-engine...origin/main   # behind 必须为 0
```

时间允许时,再跑一个最小 e2e(如 H3 T2VA 冒烟)后推送。

## 第 5 步 · 推送到 rf

```bash
git push rf paprika-engine
```

注意:

- origin 的 push 是 DISABLE 的,不会误推;
- 推送前可用 `git ls-remote --heads rf paprika-engine` 确认远端分支状态
  (是否落后、是否有他人在用)。

## 例外:什么时候改用临时试验分支

只有两种情况偏离本流程:

1. **merge 需要跨多个会话**才能解完(paprika-engine 会被锁在 mid-merge 状态);
2. **需要先在远端 CI 验证**合并结果、再让 paprika-engine 前进。

此时开 `sync/paprika-<日期>` 分支做 merge,验证通过后 `git merge --ff-only` 或
cherry-pick 该 merge commit 回 paprika-engine。其余情况一律直 merge。

## 附录:merge 前后行为差异的 A/B 对照环境(pre/post 双 venv)

merge 后出现「merge 前没这个问题」类争议时,唯一有说服力的证据是**两边代码
各自在原生环境里跑同一命令**。merge 前后代码依赖的 vllm 大版本不同(本次:
pre 需要 0.30.0,post 需要 0.33.0),**不能共用一个 venv**,也不要在主
`.venv` 里来回升降级——实测会把 15 个包连拖下来(含 torch 2.14.0+cu130 →
2.13.0,丢 +cu130 变体),且源码编译的 SageAttention 对 torch ABI 敏感,
降级后需重编。正确做法是按 `scripts/RadiantFrame/rtx5090/create_env.sh`
的配方**另建一个独立 venv**(rust/cargo 仅源码编 vllm 才需要,装 wheel 可跳过):

```bash
# ① 切到 pre-merge 锚点(editable 安装指向工作区,必须先切再装)
git checkout paprika-pre-sync-YYYYMMDD

# ② 建 pre 环境(aliyun 源;--seed 才有 pip)
export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
uv venv .venv-premerge --python 3.12 --seed
VIRTUAL_ENV=$PWD/.venv-premerge uv pip install vllm==0.30.0
VIRTUAL_ENV=$PWD/.venv-premerge uv pip install -e .

# ③ SageAttention 源码编译(nvcc 用 CUDA 13.0)
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
VIRTUAL_ENV=$PWD/.venv-premerge uv pip install \
  git+https://github.com/thu-ml/SageAttention.git --no-build-isolation

# ④ 用 pre 环境跑同一命令(deploy.py 从 PATH 解析 vllm,先 activate)
source .venv-premerge/bin/activate
python scripts/MiniMax-H3-python/pipeline.py --config <同一份 config>

# ⑤ 测完切回并重新生成版本文件(见下方坑 1)
git checkout paprika-engine
VIRTUAL_ENV=$PWD/.venv uv pip install -e .
```

实测记录(2026-11-16/17,ref2va/rtx5090,50 步):pre 5/5 轮 ~1070s/轮;
post(修复 #7745 超时回归后)5/5 轮 ~572s/轮,端到端 1.87×。对比时读 tqdm
逐步耗时(区分缓存步/计算步)+ 客户端每轮耗时,稳态取第 3 轮以后。

**踩过的坑:**

1. **editable 安装会把 `_version.py` 写进源码树**——在 pre-merge checkout 上
   `install -e .` 后,切回 paprika-engine 时 `_version.py` 留着过期版本号
   (gitignored 不随 checkout 恢复)。切回后必须从当前 checkout 重新
   `uv pip install -e .` 重新生成;
2. **单元测试可能泄漏真实 DiffusionWorker 进程**(每 rank ~3GB 显存,SIGTERM
   无效需 `kill -9`),A/B 实测前先 `nvidia-smi --query-compute-apps=pid` 检查;
3. pre-merge 代码在 post 环境(vllm 0.33)里**连 import 都过不了**(如
   `RoutedExpertsLists` 已移除),反之亦然——跑挂时先分辨是环境不匹配还是
   真回归,别急着归因;
4. 每个额外 venv 约 7GB 磁盘,定位完成后可整目录删除。

## 速查卡

```bash
git tag paprika-pre-sync-$(date +%Y%m%d)   # ① 安全网
git merge origin/main                       # ② 直 merge,冲突一次解完
git commit                                  # ③ 手写 message,点名 paprika-engine
python -c "import vllm_omni"                # ④ 冒烟
git push rf paprika-engine                  # ⑤ 推 fork
# 中途放弃:git merge --abort
# 提交后悔:git reset --hard paprika-pre-sync-YYYYMMDD
```
