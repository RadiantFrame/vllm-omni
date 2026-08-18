#!/bin/bash
# MiniMax-H3 FL2VA on 1x H800 (80 GiB HBM, Hopper SM90), host RAM ~2 TiB.
# Single-card SOTA config, adapted from deploy/pro6000/1pro6000/deploy_fp8.sh.
#
# Topology: model-level CPU offload (SequentialOffloadHook) -- the FP8 DiT
# (~31 G) and the BF16 Qwen3-VL encoder (~51.5 G) are mutually exclusive on
# the GPU; the inactive side sits in (pinned) host memory. Single GPU, so no
# TP/USP; VAE patch-parallel stays 1 (tiling on by default).
#
# Stack (each piece proven on this machine's 2h800/4h800 runs):
#   --enable-cpu-offload  : the only way H3 fits one 80G card (encoder BF16
#                           51.5G + FP8 DiT 31G = 83G > 80G if co-resident).
#   BF16 (NO --quantization): online FP8 is NOT usable on one 80G card --
#                           MEASURED (logs/h3_0818_1h800.log): the load-time
#                           BF16->FP8 conversion (scaled_fp8_quant, a CUDA
#                           kernel) runs per-layer ON GPU during
#                           _process_weights_after_loading and OOMs at ~78G
#                           (encoder 51.5G + DiT layers mid-conversion). The
#                           pro6000 FP8+offload variant only survived because
#                           that card has 96G. With FP8+DLO also documented
#                           incompatible, BF16 + cpu-offload is the only
#                           single-80G-card path. Costs: swap traffic 62G per
#                           direction (vs 31G) and no FP8 GEMM; quality is
#                           slightly BETTER (BF16 reference precision).
#   Cache-DiT "high"      : biggest single win (4xH800: 155.7->114.8s, -26%).
#   FLASH_ATTN            : = FA3 on Hopper (fa3-fwd), the measured optimum;
#                           do NOT use pro6000's CUDNN_ATTN here (Blackwell).
#   regional compile      : default, no --enforce-eager; works with sequential
#                           offload. First request pays ~30s compile warmup.
#
# vs pro6000 deploy_fp8.sh, dropped because THIS host has ~2 TiB RAM:
#   VLLM_OMNI_PIN_CPU_MEMORY=0, systemd-run MemoryMax fuse, systemd-oomd
#   dance -- all were 125 GiB-host survival gear. Keep pinned copies (faster
#   H2D); peak host usage ~79G transition + overhead is trivial here.
#
# MEASURED (2026-08-18, logs/h3_0818_1h800.log, 480p/832x480/5s, 50 steps):
#   steady 52.8s (7 consecutive 52.73-53.08s; req#1 109s = compile warmup,
#   req#2 85.8s = lazy-init settling -- exclude both). denoise ~43s (~81% of
#   E2E); the rest is encode + swaps + VAE + MP4. 58.9 GiB after model load.
#   Cross-card: 2xH800 TP2 = 26.5s -> single card is 1.99x (near-perfect halving).
# Tune: raise cache residual_diff_threshold 0.04 -> 0.10/0.20 for speed ONLY
#   after passing the quality gate (fixed-seed LPIPS/PSNR + audio cosine).
# Do NOT add --quantization fp8 here (see BF16 note above): startup OOM on
#   80G. FP8 single-card would need a 96G+ card (pro6000 path) or a code fix
#   to move the online-quant conversion off the GPU.
# Ref2VA is a separate partition: restart with MODEL=.../MiniMax-H3/Ref2VA.
# Verify in logs: "Cache-dit enabled successfully" /
#   "Resolved ... 'FLASH_ATTN'" (no "Falling back").

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800
# Post-denoise stages (DiT swap-out over PCIe + VAE decode + CPU MP4 encode)
# can exceed the engine's default 30s async-output timeout -> request aborted
# AFTER the video was actually generated.
export VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT=300

MODEL=${MODEL:-/data/models/modelscope/MiniMax/MiniMax-H3/FL2VA}
PORT=${PORT:-9000}

echo "Starting MiniMax-H3 FL2VA on 1xH800 (cpu-offload + BF16 + Cache-DiT high + FA3), port $PORT ..."

# shellcheck disable=SC2086
vllm serve "${MODEL}" \
  --omni --trust-remote-code \
  --host 0.0.0.0 --port "${PORT}" \
  --num-gpus 1 \
  --enable-cpu-offload \
  --cache-backend cache_dit \
  --cache-config '{"Fn_compute_blocks":1,"Bn_compute_blocks":0,"max_warmup_steps":4,"residual_diff_threshold":0.04,"max_continuous_cached_steps":1,"enable_taylorseer":false}' \
  --enable-cache-dit-summary \
  --diffusion-attention-backend FLASH_ATTN
