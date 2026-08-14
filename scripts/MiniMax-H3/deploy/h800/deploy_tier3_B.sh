#!/bin/bash
# Tier 3 / Candidate B -- FP8, TP4 (weight-only sharding).
# Ref: plan Tier 3 candidate B.
#
# Parallelism: TP4 (weight 4-way) + text-encoder-TP4. NO USP (no sequence split).
# Precision : FP8.
# Tradeoff  : Lowest memory (~30-45 GB/GPU) -- good headroom. But TP does NOT split
#             the sequence -> each rank runs full-length attention, and every DiT
#             layer (x50) needs 2 all-reduces -> heavier comm than USP variants.
#             Expect SLOWER denoise than A/D on long video sequences. Use this when
#             you need memory headroom (e.g. longer/higher-res), not for min latency.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800

PORT=${PORT:-8000}
NUM_WEIGHT_LOAD_THREADS=${NUM_WEIGHT_LOAD_THREADS:-8}

PROFILER_FLAGS=""
if [ "${PROFILER:-0}" = "1" ]; then
    PROFILER_FLAGS="--enable-diffusion-pipeline-profiler"
fi

echo "Starting MiniMax-H3 FL2VA Tier 3 / Candidate B (TP4+TE-TP4, FP8) on port $PORT ..."

# shellcheck disable=SC2086
vllm serve /data/models/modelscope/MiniMax/MiniMax-H3/FL2VA \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port "$PORT" \
  --num-gpus 4 \
  --num-weight-load-threads "$NUM_WEIGHT_LOAD_THREADS" \
  --tensor-parallel-size 4 \
  --text-encoder-tp-size 4 \
  --quantization fp8 \
  --diffusion-compile-granularity regional \
  --vae-patch-parallel-size 4 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --diffusion-attention-backend FLASH_ATTN \
  $PROFILER_FLAGS \
  --omni
