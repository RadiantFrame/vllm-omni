#!/bin/bash
# Tier 5 / attn_flash -- attention-backend A/B REFERENCE (FLASH_ATTN = FA3), on the
# current best config (Candidate A + Cache-DiT).
# Ref: plan Tier 5. Builds on deploy_tier4_4b.sh (Candidate A + Cache-DiT = 114.77s).
# Identical to tier4_4b; included so the attention A/B lives in one naming series.
# Control variable: --diffusion-attention-backend only (everything else, incl.
# Cache-DiT, held fixed).
#
# FLASH_ATTN resolves to FlashAttention-3 here (fa3_fwd_interface, Hopper sm90) --
# the expected winner. Use _flashhub / _flashinfer to confirm on top of Cache-DiT.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800

PORT=${PORT:-8000}
NUM_WEIGHT_LOAD_THREADS=${NUM_WEIGHT_LOAD_THREADS:-8}

PROFILER_FLAGS=""
if [ "${PROFILER:-0}" = "1" ]; then
    PROFILER_FLAGS="--enable-diffusion-pipeline-profiler"
fi

echo "Starting MiniMax-H3 FL2VA Tier 5 / attn_flash (FLASH_ATTN=FA3, +Cache-DiT) on port $PORT ..."

# shellcheck disable=SC2086
vllm serve /data/models/modelscope/MiniMax/MiniMax-H3/FL2VA \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port "$PORT" \
  --num-gpus 4 \
  --num-weight-load-threads "$NUM_WEIGHT_LOAD_THREADS" \
  --usp 4 \
  --ring 1 \
  --text-encoder-tp-size 4 \
  --quantization fp8 \
  --cache-backend cache_dit \
  --cache-config '{"Fn_compute_blocks":1,"Bn_compute_blocks":0,"max_warmup_steps":4,"residual_diff_threshold":0.04,"max_continuous_cached_steps":1,"enable_taylorseer":false}' \
  --enable-cache-dit-summary \
  --diffusion-compile-granularity regional \
  --vae-patch-parallel-size 4 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --diffusion-attention-backend FLASH_ATTN \
  $PROFILER_FLAGS \
  --omni
