#!/bin/bash
# Tier 1 deploy for MiniMax-H3 on 4xH800.
# Builds on deploy_tier0.sh (Candidate D base) by explicitly PINNING the plan's
# Tier 1 "zero-cost runtime" optimizations.
#
# IMPORTANT (overlap): Candidate D (the Tier 0 base) already ships most Tier 1 knobs
# by default -- regional torch.compile (no --enforce-eager), VAE pp4+tile, FLASH_ATTN.
# So the real command-line delta vs tier0 is small:
#   + --diffusion-compile-granularity regional   (explicit; equals the default)
#   + --num-weight-load-threads N                (explicit; tier0 leaves it at default 4)
# and the remaining Tier 1 items are automatic (no flag; see footer).
# => For H3, Tier 1 is mostly "pin + document": expect ~no INFERENCE-latency delta
#    vs tier0 (weight-load threads only speed up model load / startup). The first
#    measurable latency lever is the next tier (FP8).

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800

PORT=${PORT:-8000}
NUM_WEIGHT_LOAD_THREADS=${NUM_WEIGHT_LOAD_THREADS:-8}

# Stage profiler (text_encoder / diffuse / vae.decode). Default OFF.
PROFILER_FLAGS=""
if [ "${PROFILER:-0}" = "1" ]; then
    PROFILER_FLAGS="--enable-diffusion-pipeline-profiler"
fi

echo "Starting MiniMax-H3 FL2VA Tier 1 (Candidate D + runtime opts pinned) on port $PORT ..."

# shellcheck disable=SC2086
vllm serve /data/models/modelscope/MiniMax/MiniMax-H3/FL2VA \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port "$PORT" \
  --num-gpus 4 \
  --num-weight-load-threads "$NUM_WEIGHT_LOAD_THREADS" \
  --usp 4 \
  --ring 1 \
  --use-hsdp \
  --hsdp-shard-size 4 \
  --text-encoder-tp-size 4 \
  --diffusion-compile-granularity regional \
  --vae-patch-parallel-size 4 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --diffusion-attention-backend FLASH_ATTN \
  $PROFILER_FLAGS \
  --omni

# Tier 1 items (plan) -> flag mapping:
#   1.1 regional torch.compile        --diffusion-compile-granularity regional (and NO --enforce-eager)
#   1.2 VAE patch-parallel + tiling   --vae-patch-parallel-size 4 --vae-parallel-mode tile --vae-use-tiling
#   1.3 FLASH_ATTN backend            --diffusion-attention-backend FLASH_ATTN (= FA3 on Hopper)
#   1.4 multi-thread weight load      --num-weight-load-threads N  (startup only; does not change inference latency)
#   1.5 async diffusion output        default (step-execution off); no flag needed
# Automatic (no flag, already merged): fused RMSNorm+RoPE (#5801), video frame-conversion memory bounding (#5732)
