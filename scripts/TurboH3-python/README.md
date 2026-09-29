# TurboH3-python

LightX2V Turbo few-step LoRA students of MiniMax-H3, deployed through
the dynamic LoRA route and reusing the MiniMax-H3-python pipeline
(`pipeline.py` here is a thin import wrapper).

## What differs from MiniMax-H3-python

| Aspect | MiniMax-H3 (base) | Turbo ref2v 4-step v0.1 |
|---|---|---|
| Deploy `lora_backend` / `lora_path` | — | `peft` + the artifact file (verbatim filename — it IS the contract) |
| Deploy partition | any | `--task-type ref2va` (Ref2VA artifacts refuse combined servers) |
| Offload | optional | **off** (recipe: non-offloaded or DLO only) |
| `num_inference_steps` | 50 | **5** (4 forwards; the step scheduler counts sigma points) |
| `flow_shift` | 12 | 12 (must equal the artifact row's value) |
| Request `lora` field | — | **required** every request: `{"name","path","scale"}` |

## Artifact table (recipe summary)

| Artifact | Task | steps | flow_shift | alpha |
|---|---|---:|---:|---:|
| `minimax_h3_ref2v_turbo_4step_v0.1_bf16` | Ref2VA | 5 | 12 | 8 |
| `minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16` | Ref2VA | 9 | 6 | 8 |

`_comfyui_` exports fuse Q/K/V and are refused by name; never rename an
artifact — the filename is validated verbatim.

## Run (8x RTX 5090)

```bash
python scripts/TurboH3-python/pipeline.py \
  --config scripts/TurboH3-python/configs/ref2va/rtx5090/config.json
```

TP8 / serial shard loading / bf16 / no offload mirrors the memory
findings from the FastH3 work on 32 GB cards.
