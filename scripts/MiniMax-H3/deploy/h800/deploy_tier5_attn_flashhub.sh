#!/bin/bash
# Tier 5 / attn_flashhub -- attention-backend A/B variant: FLASH_ATTN_3_HUB (HF FA3),
# on the current best config (Candidate A + Cache-DiT).
# Ref: plan Tier 5. Builds on deploy_tier4_4b.sh (= 114.77s).
# Control variable: --diffusion-attention-backend only (FLASH_ATTN -> FLASH_ATTN_3_HUB);
# Cache-DiT and all else held fixed.
#
# FLASH_ATTN_3_HUB = HuggingFace kernels flash-attn3 = HF-hosted FA3 (requires
#   sm>=9.0; H800 qualifies). Pulls kernels-community/flash-attn3 via the HF `kernels`
#   lib -> `pip install kernels` first, else it silently falls back to local FLASH_ATTN.
#   A/B vs attn_flash (local FA3 via fa3-fwd): both are FA3 -> expect parity-ish, pick
#   faster. (Note: FLASH_ATTN_HUB without the "3" = HF FA2, slower -- not used here.)

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800

PORT=${PORT:-8000}
NUM_WEIGHT_LOAD_THREADS=${NUM_WEIGHT_LOAD_THREADS:-8}

PROFILER_FLAGS=""
if [ "${PROFILER:-0}" = "1" ]; then
    PROFILER_FLAGS="--enable-diffusion-pipeline-profiler"
fi

echo "Starting MiniMax-H3 FL2VA Tier 5 / attn_flashhub (FLASH_ATTN_3_HUB = HF FA3, +Cache-DiT) on port $PORT ..."

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
  --diffusion-attention-backend FLASH_ATTN_3_HUB \
  $PROFILER_FLAGS \
  --omni
