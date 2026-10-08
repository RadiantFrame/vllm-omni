# TurboH3-python

LightX2V Turbo few-step LoRA students of MiniMax-H3, deployed through
the dynamic LoRA route and reusing the MiniMax-H3-python pipeline
(`pipeline.py` here is a thin import wrapper).

## What differs from MiniMax-H3-python

| Aspect | MiniMax-H3 (base) | Turbo 4-step v0.1 |
|---|---|---|
| Deploy `lora_backend` / `lora_path` | — | `peft` + the artifact file (verbatim filename — it IS the contract) |
| Deploy partition | any | fl2v artifacts: `--task-type fl2va` (serves t2va+fl2va); ref2v artifacts demand `--task-type ref2va` |
| Offload | optional | DLO (ref2va preset) or none (light packings); plain CPU offload is refused for Turbo |
| `num_inference_steps` | 50 | **4** (Omni counts transformer forwards; the recipe table's 5 is FastVideo's sigma-node counting) |
| `flow_shift` | 12 | 12 (must equal the artifact row's value) |
| Request `lora` field | — | **required** every request: `{"name","path","scale"}` |

## Artifact support matrix (all eight supported Diffusers artifacts, steps in Omni convention)

| Group | Artifact | Task | steps | flow_shift | alpha | Status |
|---|---|---|---:|---:|---:|---|
| fl2v 4-step | `minimax_h3_fl2v_turbo_4step_v0.1.safetensors` | T2VA / FL2VA | 4 | 12 | none -> 8 fallback | **in use: t2va, fl2va presets** |
| fl2v 4-step | `minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16` | T2VA / FL2VA | 4 | 6 | 128 | local, untried |
| fl2v 4-step | `minimax_h3_fl2v_turbo_4step_v1.1_768p_bf16` | T2VA / FL2VA | 4 | 6 | 128 | local, untried |
| fl2v 4-step | `minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16` | T2VA / FL2VA | 4 | 6 | 8 | local, untried |
| fl2v 8-step | `minimax_h3_fl2v_turbo_8step_v1.0_bf16` | T2VA / FL2VA | 8 | 12 | 8 | local, untried |
| fl2v 8-step | `minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16` | T2VA / FL2VA | 8 | 6 | 8 | local, untried |
| ref2v 4-step | `minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors` | Ref2VA | 4 | 12 | 8 | **in use: ref2va preset** |
| ref2v 8-step | `minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16` | Ref2VA | 8 | 6 | 8 | quality dial (below) |

All eight are in the local `/data/models/modelscope/lightx2v/Minimax-h3-Turbo`
download, so future A/Bs need no downloads. The local directory also
holds an fp8 export of fl2v v1.1 that is NOT among the supported eight.

Never rename an artifact — the filename is validated verbatim. v0.1
artifacts declare no alpha: the server falls back to 8 with a warning
(expected noise).

### Future quality dial: the 8-step v1.0_768p artifacts

The presets all run 4-step v0.1 (the speed baseline). The 8-step
artifacts — e.g. `minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16`, already
in the local Turbo download — double the denoiser evaluations
(`num_inference_steps=8`, `flow_shift=6`, alpha properly declared) and
are trained specifically for 768p, our output tier. Untested here; a
natural A/B is quality vs ~2x denoise time (ref2va 218 s -> roughly
370-400 s, still ~3x faster than base) before adopting it for
quality-sensitive work.

## Measured performance (8x RTX 5090, all 5/5 rounds zero-error)

| Preset | LoRA artifact | Profile | Output | Steady e2e | vs base 50-step |
|---|---|---|---|---|---|
| `configs/t2va/rtx5090` | `minimax_h3_fl2v_turbo_4step_v0.1` | TP4/USP2 · fp8 · SAGE · eager, no offload | 1344x768 / 124 f / 24 fps | **~15.4 s** | ~142 s -> **9.2x** |
| `configs/fl2va/rtx5090` | `minimax_h3_fl2v_turbo_4step_v0.1` | TP4/USP2 · fp8 · SAGE · eager, no offload | 1344x768 / 124 f / 24 fps | **~17.1 s** | ~56 s -> **3.3x** |
| `configs/ref2va/rtx5090` | `minimax_h3_ref2v_turbo_4step_v0.1_bf16` | TP4/USP2 · fp8 · SAGE · eager · **DLO** | 1344x768 / 362 f / 24 fps | **~218 s** | 1072 s -> **4.9x** |

Log evidence: `logs/20261007-184350` (t2va), `logs/20261007-183025`
(fl2va), `logs/20261007-185128` (ref2va).

ref2va tuning ladder (same 15 s input, each step measured): base
1072 s -> 478 s (Turbo + DLO + CUDNN) -> 316 s (SAGE, after the torch
2.13 bmm-ABI fix) -> 218 s (TP4/USP2). TP4 beats TP8 even on the
226k-token packing once DLO+SAGE remove the residency constraint —
per-layer collectives dominate; "needs TP8" only held without weight
streaming.

## Run (8x RTX 5090)

```bash
python scripts/TurboH3-python/pipeline.py \
  --config scripts/TurboH3-python/configs/<t2va|fl2va|ref2va>/rtx5090/config.json
```

Common profile: fp8 weights, SAGE_ATTN (torch 2.13.0+cu130 pinned —
downgrading re-breaks the bmm ABI), serial shard loading
(`num_weight_load_threads: 1`), `enforce_eager` (regional compile's
first request deterministically OOMs on these memory-tight profiles),
DLO only where the packing needs weight streaming (ref2va).
