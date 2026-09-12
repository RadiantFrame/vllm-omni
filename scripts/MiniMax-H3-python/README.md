# MiniMax-H3 Python Toolbox (FL2VA)

Python toolchain for benchmarking MiniMax-H3 FL2VA video generation on
vLLM-Omni: deploy the service, fan out generation traffic, extract
performance metrics, and grid-search the configuration space. It is the
Python counterpart of the bash scripts under `scripts/MiniMax-H3/` and
`scripts/MiniMax-H3-Ref2VA/`, refactored into composable classes.

## Modules

| File | Class | Role |
|---|---|---|
| `deploy.py` | `DeployConfig` / `Deployer` | Start/stop one `vllm serve` service and wait for health |
| `generate.py` | `GenerateConfig` / `Generator` | Fan out N rounds of FL2VA requests to service port(s) |
| `metrics.py` | `Metrics` / `LogParser` | Extract steady-state metrics from a service log |
| `pipeline.py` | `PipelineConfig` / `Pipeline` | One run: deploy → generate → stop → collect metrics |
| `search.py` | `SearchConfig` / `Search` | Grid search: many Pipeline runs + a flat JSONL index |

Dependency chain (each layer imports only below it):

```
deploy.py   generate.py      atomic capabilities (know nothing of each other)
      \       /
     metrics.py              run artifacts → metrics (LogParser + future sources)
        \    |
      pipeline.py            one run = deploy + generate + metrics
          |
        search.py            many runs = grid expansion + loop + index
```

The only cross-module coupling is duck-typed: `Generator.from_deployer(d)`
just reads `d.port`, so generate.py stays independent of deploy.py.

## Quick start

### One run (deploy + 5-round traffic + metrics)

```bash
cd scripts/MiniMax-H3-python
python pipeline.py                    # baseline run from env knobs
```

Every run owns a timestamped directory under `logs/`:

```
logs/20260904-153012/
  config.json     effective-config snapshot + meta (started / git / host / GPUs)
  deploy.log      the vllm serve stdout+stderr for this run
  metrics.json    Metrics summary (steady-state e2e, cache-dit, failures)
  outputs/        generated .mp4 videos
```

Directory names carry only WHEN (timestamp); parameters live in
`config.json`. Treat `logs/` as a queryable library of runs:

```bash
jq -c 'select(.deploy.usp==2)' logs/*/config.json     # which runs used usp=2
python metrics.py logs/20260904-153012/deploy.log      # re-parse any run
```

### Grid search

```python
# scripts/MiniMax-H3-python$ python - <<'EOF'
from deploy import DeployConfig, DEFAULT_CACHE_CONFIG
from generate import GenerateConfig
from pipeline import PipelineConfig
from search import SearchConfig, Search

cfg = SearchConfig(
    pipeline_base=PipelineConfig(
        deploy_base=DeployConfig(cuda_visible_devices="4,5,6,7"),
        generate_base=GenerateConfig(duration=15),
    ),
    grid={
        "usp": [1, 2],
        "cache_config.residual_diff_threshold": [0.04, 0.06],   # dotted = deep-merge
        "cache_config.enable_taylorseer": [False, True],
    },
)
Search(cfg).run()          # one Pipeline run per grid point
# EOF
```

Grid axes are DeployConfig/GenerateConfig field names, dispatched
automatically by name (a parameter is never declared twice). Dict fields
are swept via dotted paths that deep-merge, so only the named sub-key
varies. Results append to `logs/index.jsonl` (one row per run, pointing at
its run directory — a derived cache; each run stays self-describing via its
own config.json/metrics.json).

### Standalone pieces

```bash
# service lifecycle (parity with the bash deploy scripts)
python deploy.py --dry-run       # print the vllm serve command
python deploy.py --detach        # background; pid -> logs/deploy.pid
python deploy.py --stop          # graceful stop via pid file (TERM→wait→KILL→verify)

# traffic against an already-running service
python generate.py               # INPUT_DIR + env knobs, 5 rounds

# metrics from any service log
python metrics.py logs/<run>/deploy.log [--json] [--warmup 2]
```

## Configuration

Each `*Config` dataclass mirrors its env knobs (uppercased field names) via
`from_env()`, and can equally be constructed/overridden in Python —
`dataclasses.replace()` re-runs `__init__`/`__post_init__`, so derived
fields (INPUT_DIR resolution, run_dir timestamps) stay consistent.

- **`DeployConfig`** (deploy.py): model, port, `CUDA_VISIBLE_DEVICES`,
  parallelism (TP/USP/RING/text-encoder TP/VAE patch), quantization,
  cpu-offload, and `cache_config` — a dict whose keys mirror vLLM-Omni's
  official `DiffusionCacheConfig` (unknown keys are rejected; the
  `CACHE_CONFIG` env var accepts a JSON override merged on top).
- **`GenerateConfig`** (generate.py): shape (`task_type`, `width`/`height` or
  `aspect_ratio` — one of 21:9/16:9/4:3/1:1/3:4/9:16, which replaces both and
  lets the server derive the 768-short-edge canvas; `adaptive`/`auto` = server
  default, `duration`, `seed`), fan-out (`rounds`, `ports`, `host`, `out_dir`), and
  `INPUT_DIR` — the only input knob. A case directory holds `prompt.txt`
  plus 0–2 reference frames (0 = text-only, 1 = first frame, 2 = first +
  last frame, sorted filename order = upload order; no URL downloads).
- **`PipelineConfig`** (pipeline.py): the two bases + `warmup` (leading
  requests Metrics drops) + `run_dir` (auto: `logs/<YYYYmmdd-HHMMSS>`).
- **`SearchConfig`** (search.py): `pipeline_base`, `grid`, `index_path`.

## Metrics extracted per run

`Metrics.collect()` (see `metrics.json` in any run dir; sourced today from
the service log via `LogParser` — visual-quality families over
`outputs/*.mp4` plug in here later):

- **e2e_total_ms** steady mean/median/min/max + per-round series (rounds
  1–2 are compile warmup / lazy-init settling and are excluded by default)
- attribution: `denoise_step_latency_ms`,
  `diffusion_engine_{exec,total}_time_s`, `postprocess_time_s`
- cache-dit: executed vs transformer steps, residual-diff percentiles
  (requires `--enable-cache-dit-summary`, on by default in DeployConfig)
- resources: model-load GiB/seconds, per-worker GPU memory
- sanity: request shape (resolution / steps / image count / audio duration)
- failures: FATAL / Traceback / OOM / health-timeout count (nonzero →
  `metrics.json` `failures > 0` and CLI exit code 1)

## Design notes

- **Config classes are the single source of truth.** `build_cmd()` /
  `build_form()` translate a config into the `vllm serve` argv / multipart
  form exactly once; CLI and library users share them.
- **Graceful stop is mandatory.** Hard-killing a rank mid-collective
  orphans NCCL spin kernels that peg GPUs at 100% until a driver reset.
  All stop paths (`Deployer.stop`, `--stop`, Ctrl-C) run TERM → wait →
  KILL as last resort → verify no `DiffusionWorker` remnants.
- **Runs are self-describing and isolated.** Artifacts never leak to
  global paths; `config.json` records the complete effective config
  (including defaults), so any run is reproducible from its directory
  alone.
- **The search index is derived data.** Deleting `logs/index.jsonl` loses
  nothing — it can be rebuilt from the per-run JSONs.
