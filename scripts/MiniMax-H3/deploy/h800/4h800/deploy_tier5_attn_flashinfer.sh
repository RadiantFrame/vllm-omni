#!/bin/bash
# Tier 5 / attn_flashinfer -- attention-backend A/B variant: FLASHINFER_ATTN, on the
# current best config (Candidate A + Cache-DiT).
# Ref: plan Tier 5. Builds on deploy_tier4_4b.sh (= 114.77s).
# Control variable: --diffusion-attention-backend only (FLASH_ATTN -> FLASHINFER_ATTN);
# Cache-DiT and all else held fixed.
#
# FlashInfer backend. A/B vs attn_flash (FA3) on `diffuse`, both with Cache-DiT.
# Needs the flashinfer package; if missing, startup will fall back / error -- check
# the "Resolved diffusion attention backend ..." log line.
#
# ⚠️ KNOWN-INCOMPATIBLE WITH H3 (do not use): H3 uses a packed multimodal custom
#   attention mask; FlashInfer's BatchPrefillWithCustomMask.plan() -> segment_packbits
#   overflows and crashes on the first request:
#     RuntimeError: Trying to create tensor with negative dimension -70891520
#   (flashinfer/prefill.py + quantization/packbits.py). Local FLASH_ATTN (FA3) handles
#   this mask fine and is the optimal/working backend for H3. This script is kept only
#   to document the incompatibility.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800

PORT=${PORT:-8000}
NUM_WEIGHT_LOAD_THREADS=${NUM_WEIGHT_LOAD_THREADS:-8}

PROFILER_FLAGS=""
if [ "${PROFILER:-0}" = "1" ]; then
    PROFILER_FLAGS="--enable-diffusion-pipeline-profiler"
fi

echo "Starting MiniMax-H3 FL2VA Tier 5 / attn_flashinfer (FLASHINFER_ATTN, +Cache-DiT) on port $PORT ..."

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
  --diffusion-attention-backend FLASHINFER_ATTN \
  $PROFILER_FLAGS \
  --omni
