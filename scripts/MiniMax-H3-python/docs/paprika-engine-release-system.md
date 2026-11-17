# paprika-engine 发版体系

> fork(RadiantFrame/vllm-omni)的主线发版与版本推导规则。记录日期:2026-11-17

## 拓扑总览

```
分支:  paprika-engine            ← fork 默认分支,自标签(打本仓库的版本 tag)
tag:   paprika-engine/vX.Y.Z     ← 同名命名空间,记录"主线的发布"
上游:  merge origin/main → paprika-engine(流程见 merge-main-into-paprika-engine.md)
topic: feat/* 从 paprika-engine 切出、合回、删除
```

## 1. 主线分支:paprika-engine

- fork(`rf` 远端)的默认分支,所有发布都从这里产生;
- 吸收上游的方式是**直接在本分支上 merge origin/main**(安全网 tag + 手写
  merge message,完整流程见 [merge-main-into-paprika-engine.md](merge-main-into-paprika-engine.md)),
  不开牺牲性的 sync/* 分支;
- origin(官方 vllm-project/vllm-omni)只读,推送一律走 `git push rf paprika-engine`。

## 2. 版本 tag:paprika-engine/vX.Y.Z

- **同名命名空间**:`paprika-engine/` 前缀标记"这是 fork 主线的发布",
  与上游的 `v0.X.Y` 官方 tag 一眼可分;
- 打在 paprika-engine 上经过验证的版本点(功能齐 + 测试过 + 基准记录在
  run_dir/metrics.json),例如:
  `v0.1.0` rtx5090 presets → `v0.1.1` export_serve → `v0.1.2` 8-step 档
  → (merge + 1.89× 验证后) `v0.2.0`;
- 另有非发版锚点 tag(`paprika-pre-sync-YYYYMMDD` / `jw-pre-sync-*`),
  只作 merge 前回滚安全网,不算发版。

## 3. 版本号推导与隔离(为什么 paprika tag 不污染 `__version__`)

- `pyproject.toml` 的 `[tool.setuptools_scm]` **无显式配置**,依赖默认
  `tag_regex`:只认 `v` 开头的纯版本形态 tag(如 `v0.33.0rc1`);
- `paprika-engine/v0.1.2`、`paprika-pre-sync-*` 不符合版本形态,**天然被
  排除在版本推导之外**——fork 自己的 tag 再密也不会改变
  `vllm_omni.__version__`;
- 实证(2026-11-17):`git describe --tags` 显示
  `paprika-pre-sync-20261116-1085-g...`(describe 看所有 tag),而
  setuptools_scm 推导出 `0.33.0rc2.dev418`(只看版本 tag,最近的是
  `v0.33.0rc1`,距 HEAD 418 个 commit);
- 因此版本号只反映**上游基线**:merge main 之后要 `git fetch origin --tags`
  更新锚点,否则 `__version__` 停在旧上游 tag(详见 merge 文档第 0 步)。

## 4. topic 分支:feat/*

```
git checkout -b feat/<topic> paprika-engine   # 切出
# ...开发、自测...
git checkout paprika-engine && git merge feat/<topic>   # 合回
git branch -d feat/<topic> && git push rf --delete feat/<topic>  # 删除
```

- 生命周期短:切出 → 合回 → 删除,不留长期平行分支;
- 不从 topic 直接对 origin/main;上游吸收永远只发生在 paprika-engine 的
  merge 上。

## 速查

```bash
git tag paprika-engine/vX.Y.Z              # 发版(在验证过的 paprika-engine 版本点)
git tag paprika-pre-sync-$(date +%Y%m%d)   # merge 前安全网(非发版)
git describe --tags --match 'v*'           # 查当前上游版本锚点(供 setuptools_scm)
git push rf paprika-engine --follow-tags   # 推分支并带上发版 tag
```
