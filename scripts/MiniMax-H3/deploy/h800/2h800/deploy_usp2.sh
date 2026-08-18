#!/bin/bash
# EXPERIMENTAL (2xH800) -- USP2 variant of deploy.sh (README 调参入口 #3).
# Same stack as deploy.sh (FP8 + Cache-DiT high + VAE pp2 + FA3 + regional
# compile) but swaps TP2 -> USP2 for the DiT: sequence 2-way sharding instead
# of weight sharding. Everything else held fixed -- single control variable.
#
# ⚠️ OOM RISK -- USP does NOT shard weights: the FULL FP8 DiT (~33GB) stays
#   replicated per card + encoder shard (~25.7GB) ~= 65GB model resident
#   (vs ~50GB with TP2), before compile workspace + activations. Expected to
#   survive 480p/832x480/5s; may OOM at 768p/1344x768/8s. If it OOMs: drop
#   resolution/duration first (e.g. 480p/5s workload), then give up.
# Why try: USP2 avoids TP's per-layer all-reduce (50 layers x 2 collectives)
#   and is the shape that won on 4xH800 (USP4). Measure denoise s/it vs
#   deploy.sh on the SAME workload; keep whichever is faster AND fits.
# Validity: ulysses=2 divides 56 heads (28/card) -- OK; TE-TP2 = world size.
#
# RESULT (2026-08-18, logs/h3_0818_2h800.log, 480p/832x480/5s, 50 steps):
#   USP2 = 29.1s steady / 0.58s/it / 65.8 GiB model per card (no OOM).
#   TP2  = 26.5s steady / 0.53s/it / 50.4 GiB per card.
#   -> TP2 WINS by ~9.8% and uses 15GB less. Keep deploy.sh as the 2-GPU
#   recommendation; this script is kept to document the negative result.
#
# RESULT (2026-08-18, logs/h3_0818_2h800_768p.log, 768p/1344x768/8s):
#   USP2 = OOM. Model resident 67.2 GiB/card; request #1 barely finished
#   (229.8s incl. compile warmup), request #2 died with CUDA OOM
#   (78.79 GiB in use, 418 MiB allocation failed in the Ulysses all-to-all
#   output reshape). USP2 on 2xH800 is 480p-only; 768p requires deploy.sh (TP2).

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800

PORT=${PORT:-9000}
NUM_WEIGHT_LOAD_THREADS=${NUM_WEIGHT_LOAD_THREADS:-8}

PROFILER_FLAGS=""
if [ "${PROFILER:-0}" = "1" ]; then
    PROFILER_FLAGS="--enable-diffusion-pipeline-profiler"
fi

echo "Starting MiniMax-H3 FL2VA EXPERIMENTAL USP2 (+FP8 + Cache-DiT high) on 2xH800, port $PORT ..."

# shellcheck disable=SC2086
vllm serve /data/models/modelscope/MiniMax/MiniMax-H3/FL2VA \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port "$PORT" \
  --num-gpus 2 \
  --num-weight-load-threads "$NUM_WEIGHT_LOAD_THREADS" \
  --usp 2 \
  --ring 1 \
  --text-encoder-tp-size 2 \
  --quantization fp8 \
  --cache-backend cache_dit \
  --cache-config '{"Fn_compute_blocks":1,"Bn_compute_blocks":0,"max_warmup_steps":4,"residual_diff_threshold":0.04,"max_continuous_cached_steps":1,"enable_taylorseer":false}' \
  --enable-cache-dit-summary \
  --diffusion-compile-granularity regional \
  --vae-patch-parallel-size 2 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --diffusion-attention-backend FLASH_ATTN \
  $PROFILER_FLAGS \
  --omni
