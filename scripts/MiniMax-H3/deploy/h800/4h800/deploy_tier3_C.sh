#!/bin/bash
# Tier 3 / Candidate C -- FP8, TP2 + USP2 (balanced weight+sequence sharding).
# Ref: plan Tier 3 candidate C (sglang --tp-size 2 --ulysses-degree 2 mirror).
#
# Parallelism: TP2 (weight 2-way) + USP2 (sequence 2-way) + text-encoder-TP4.
# Precision : FP8.
# NOTE on text-encoder-tp-size: the plan literally says 2, but an INTERMEDIATE
#   text_encoder_tp_size (2 on 4 GPUs) hits the cpu_group AssertionError in
#   GroupCoordinator -> only 1 and world_size are runnable. So this uses
#   --text-encoder-tp-size 4 (validated; same as your deploy.sh). The DiT
#   weight+sequence split still matches sglang's tp2+ulysses2; only the encoder
#   sharding differs (4-way vs implicit 2-way) -- no effect on output.
# Tradeoff: middle ground on both memory (~45-55 GB/GPU) and comm. A/B it against
#   A and D for the best FL2VA latency.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800

PORT=${PORT:-8000}
NUM_WEIGHT_LOAD_THREADS=${NUM_WEIGHT_LOAD_THREADS:-8}

PROFILER_FLAGS=""
if [ "${PROFILER:-0}" = "1" ]; then
    PROFILER_FLAGS="--enable-diffusion-pipeline-profiler"
fi

echo "Starting MiniMax-H3 FL2VA Tier 3 / Candidate C (TP2+USP2+TE-TP4, FP8) on port $PORT ..."

# shellcheck disable=SC2086
vllm serve /data/models/modelscope/MiniMax/MiniMax-H3/FL2VA \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port "$PORT" \
  --num-gpus 4 \
  --num-weight-load-threads "$NUM_WEIGHT_LOAD_THREADS" \
  --tensor-parallel-size 2 \
  --usp 2 \
  --text-encoder-tp-size 4 \
  --quantization fp8 \
  --diffusion-compile-granularity regional \
  --vae-patch-parallel-size 4 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --diffusion-attention-backend FLASH_ATTN \
  $PROFILER_FLAGS \
  --omni
