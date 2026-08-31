#!/bin/bash
# Tier 4 / 4b -- Cache-DiT (DBCache, H3 "high" profile) stacked on Candidate A.
# Ref: plan Tier 4 (4b). Builds on deploy_tier3_A.sh
#      (USP4 + TE-TP4 + FP8 = 155.74s steady-state).
#
# Cache: --cache-backend cache_dit, H3 "high" conservative profile
#        (residual_diff_threshold=0.04, max_warmup_steps=4, max_continuous_cached_steps=1).
#        Validated on H200 at 1.35x / SSIM 0.9709 / PSNR 34.98 dB vs lossless.
#        For more speed: raise residual_diff_threshold to 0.20-0.30 (re-run quality
#        gate). For aggressive step-skip add "scm_steps_mask_policy":"medium".
# --enable-cache-dit-summary logs per-step hit/skip so you can confirm the speedup.
# Mutually exclusive with TeaCache (deploy_tier4_4a.sh) -- run ONE, not both.
# TaylorSeer stays OFF (not suitable for distilled models).

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=4500

PORT=${PORT:-9000}
NUM_WEIGHT_LOAD_THREADS=${NUM_WEIGHT_LOAD_THREADS:-8}
MODEL=${MODEL:-/data/models/modelscope/MiniMax/MiniMax-H3/Ref2VA}
# auto: partition inferred from MODEL path (FL2VA→fl2va, Ref2VA→ref2va, root→combined)
TASK_TYPE=${TASK_TYPE:-auto}

PROFILER_FLAGS=""
if [ "${PROFILER:-0}" = "1" ]; then
    PROFILER_FLAGS="--enable-diffusion-pipeline-profiler"
fi

echo "Starting MiniMax-H3 FL2VA Tier 4 / 4b (Candidate A + Cache-DiT high) on port $PORT ..."

# shellcheck disable=SC2086
vllm serve ${MODEL} \
  --omni \
  --task-type ${TASK_TYPE} \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port "$PORT" \
  --num-gpus 4 \
  --usp 4 \
  --ring 1 \
  --text-encoder-tp-size 4 \
  --vae-patch-parallel-size 4 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --num-weight-load-threads "$NUM_WEIGHT_LOAD_THREADS" \
  --diffusion-compile-granularity regional \
  --diffusion-attention-backend FLASH_ATTN \
  --quantization fp8 \
  --cache-backend cache_dit \
  --cache-config '{"Fn_compute_blocks":1,"Bn_compute_blocks":0,"max_warmup_steps":4,"residual_diff_threshold":0.04,"max_continuous_cached_steps":1,"enable_taylorseer":false}' \
  --enable-cache-dit-summary \
  $PROFILER_FLAGS
