#!/bin/bash
# Tier 5 / kvcache_fp8_skip45 -- diffusion KV-cache FP8, CONSERVATIVE, on the
# current best config (Candidate A + Cache-DiT).
# Ref: plan Tier 5. Builds on deploy_tier4_4b.sh (= 114.77s).
# Control variable: KV-cache dtype + how many early steps/layers stay BF16;
#   Cache-DiT and all else held fixed.
#
# --diffusion-kv-cache-dtype fp8 : cast attention K/V to FP8 (mainly a MEMORY lever;
#   latency effect modest). Orthogonal to weight FP8.
# skip-steps 0-4, skip-layers 0,1 : first 5 steps and first 2 DiT blocks keep BF16
#   K/V to protect quality. Most conservative of the 3 KV-cache variants.
# NOTE: KV-cache FP8 x Cache-DiT is a less-validated combination (cache controls
#   WHICH steps compute; KV fp8 controls HOW computed attention runs -- meant to be
#   orthogonal, but verify). Quality gate REQUIRED vs attn_flash (BF16 KV).

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800

PORT=${PORT:-8000}
NUM_WEIGHT_LOAD_THREADS=${NUM_WEIGHT_LOAD_THREADS:-8}

PROFILER_FLAGS=""
if [ "${PROFILER:-0}" = "1" ]; then
    PROFILER_FLAGS="--enable-diffusion-pipeline-profiler"
fi

echo "Starting MiniMax-H3 FL2VA Tier 5 / kvcache_fp8_skip45 (conservative, +Cache-DiT) on port $PORT ..."

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
  --diffusion-kv-cache-dtype fp8 \
  --diffusion-kv-cache-skip-steps '0-4' \
  --diffusion-kv-cache-skip-layers '0,1' \
  --diffusion-compile-granularity regional \
  --vae-patch-parallel-size 4 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --diffusion-attention-backend FLASH_ATTN \
  $PROFILER_FLAGS \
  --omni
