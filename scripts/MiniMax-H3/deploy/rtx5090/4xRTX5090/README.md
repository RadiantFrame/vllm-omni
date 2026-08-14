# MiniMax-H3 FL2VA on 4× RTX 5090 (32 GiB each), single service

One H3 FL2VA service spanning all 4 GPUs. This is **not** the multi-service
fan-out in the parent `rtx5090/` dir — it is one service, TP4.

## Scripts

| Script | Status | Use |
|---|---|---|
| `deploy.sh` | ✅ works | **Primary.** TP4 + DLO(no-AllGather) + BF16 + Cache-DiT. The only config verified to fit 32 GiB cards. |
| `deploy_fp8.sh` | ❌ OOM on 5090 | TP4 + online FP8 + no DLO + regional compile. Fits only on ≥80 GiB cards; **does NOT fit 4×5090** (see below). Kept for bigger-card hosts. |

Request scripts are **hardware-independent** HTTP clients and live in the shared
[`scripts/MiniMax-H3/generate/`](../../../generate/) folder (`generate.sh`
for 1344×768, `generate_480p_5s.sh` for 832×480/5s, plus the multi-port
fan-out variant). Default target is `http://localhost:8000`; override with
`BASE_URL=...` / `PORTS="..."`.

## Empirical findings (2026-08-14, RTX 5090 32 GiB)

### `deploy.sh` (DLO + BF16) — works
- Boots cleanly, all 4 workers init, serves 480p requests.
- Peak HBM **~10 GiB/card** (DLO offloads most weights to host RAM; GPU holds
  only resident DiT blocks + buffers). ~22 GiB/card idle.
- Host RAM per service ~187 GiB (pinned weights) — fits the 503 GiB box easily
  for a single service.
- Caveat: 480p is compute-bound — raising `--dlo-resident-layers` (to use the
  idle GPU memory) gave **no single-request latency gain** in 2-card tests
  (H2D already hidden by compute overlap). The idle memory is better spent on
  concurrency / larger shapes, not single-request speed.
- Single-request speed lever on this path = Cache-DiT threshold
  (`residual_diff_threshold` 0.04 → 0.20, ~16% faster, re-check quality).

### `deploy_fp8.sh` (online FP8) — OOM, do not use on 5090
- FP8 kernel itself is fine on 5090 (Blackwell): log shows
  `CutlassFP8ScaledMMLinearKernel` + `DeepGEMM E8M0 enabled` — Gate 1 passes.
- **OOM at load-time quantization** (`fp8.py:159 scaled_fp8_quant`):
  online FP8 loads BF16 weights to GPU, then converts to FP8 per layer, so the
  BF16 original + FP8 output coexist transiently. At TP4 the BF16 model alone
  is ~38.8 GiB/card (DiT 15.5 + encoder 12.9 + VAE 10.4) > 31.4 GiB → OOM.
- No DLO ⇒ encoder/VAE on-demand staging does not fire ⇒ everything tries to
  go resident. Not fixable by tuning; it is a capacity limit.
- Error signature: `torch.OutOfMemoryError ... 30.53 GiB allocated` during
  `process_weights_after_loading`.

## Why this topology (no validated 4×5090 reference exists in-repo)

- BF16 must use **TP4** (shards DiT 4-way to ~15.5 GiB/rank); TP2 BF16 is
  ~38.8 GiB/rank and won't fit. TP4 forces `--usp 1` (DiT group is full).
- `--text-encoder-tp-size 4`: the Qwen3-VL encoder (~51.5 GiB BF16) is the
  memory hotspot; TE-TP4 shards it (64/8 heads divisible by 4).
- `--vae-patch-parallel-size 4` must equal the DiT group (4); H3 supports only
  native `tile` mode.
- `--usp 4 --tp 1` is an anti-pattern (replicates the full DiT on every card).
- FP8 ⟂ DLO (runtime weight-stride conflict), so the two configs are exclusive.

## Future direction: offline FP8 + DLO (the only way to get FP8 speed on 5090)

A **pre-quantized** FP8 checkpoint (DiT only) would:
- avoid the load-time BF16→FP8 2× transient (weights are already FP8 on disk),
- be **compatible with DLO** (`uses_meta_device=False` ⇒ passes the DLO gate),
- halve DiT weight volume under DLO and add ~12% speed.

Requires building the offline-FP8 checkpoint first (no public one exists).
The text encoder is not quantized by the H3 FP8 path (custom module, no
`quant_method`) — see prior discussion; the DiT (~62 GiB → ~31 GiB) is the
only part worth converting.

## Pointers

- Validated 2×5090 DLO baseline: `recipes/MiniMaxAI/MiniMax-H3-5090.md`
- 4-GPU topology taxonomy (H800, 80 GiB): `scripts/MiniMax-H3/h800/`
- 96 GiB RTX PRO 6000 (BF16 TP2, no offload needed): `recipes/MiniMaxAI/MiniMax-H3-RTX-PRO-6000.md`
- Design notes / decision trace: `/home/test001/.claude/plans/4-rtx5090-h3-adaptive-swan.md`
