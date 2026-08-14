#!/bin/bash
# Tier 3 / Candidate D -- official 4xH100(Hopper) perf baseline, BF16.
# Ref: plan Tier 3 candidate D; tests/dfx/perf/tests/test_minimax_h3_vllm_omni.json (#5836).
#
# Parallelism: USP4 (sequence 4-way) + HSDP4 (weight 4-way) + text-encoder-TP4.
# Precision : BF16 (NO --quantization). HSDP4 shards the full DiT so it fits 80GB
#             without FP8 -- best output quality. Peak ~54-55 GB/GPU (H100 measured
#             FL2VA 38.1s/8-step). First compile = warmup; don't count request #1.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800

PORT=${PORT:-8000}
NUM_WEIGHT_LOAD_THREADS=${NUM_WEIGHT_LOAD_THREADS:-8}

PROFILER_FLAGS=""
if [ "${PROFILER:-0}" = "1" ]; then
    PROFILER_FLAGS="--enable-diffusion-pipeline-profiler"
fi

echo "Starting MiniMax-H3 FL2VA Tier 3 / Candidate D (USP4+HSDP4+TE-TP4, BF16) on port $PORT ..."

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
