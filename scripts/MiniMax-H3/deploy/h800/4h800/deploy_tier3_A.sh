#!/bin/bash
# Tier 3 / Candidate A -- FP8, USP4 (sequence-only sharding).
# Ref: plan Tier 3 candidate A. (Same shape as deploy_tier2.sh.)
#
# Parallelism: USP4 (sequence 4-way) + text-encoder-TP4. NO HSDP, NO TP.
# Precision : FP8 (online W8A8 of DiT linears). USP replicates the full DiT per rank,
#             so FP8 halves DiT weights (66->33GB) to fit. Validated (~55-56 GB/GPU;
#             denoise ~3.1s/step, ~12% faster than BF16). FP8+HSDP is unvalidated on
#             H3 -> keep them separate (don't add --use-hsdp here).

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800

PORT=${PORT:-8000}
NUM_WEIGHT_LOAD_THREADS=${NUM_WEIGHT_LOAD_THREADS:-8}

PROFILER_FLAGS=""
if [ "${PROFILER:-0}" = "1" ]; then
    PROFILER_FLAGS="--enable-diffusion-pipeline-profiler"
fi

echo "Starting MiniMax-H3 FL2VA Tier 3 / Candidate A (USP4+TE-TP4, FP8) on port $PORT ..."

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
  --diffusion-compile-granularity regional \
  --vae-patch-parallel-size 4 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --diffusion-attention-backend FLASH_ATTN \
  $PROFILER_FLAGS \
  --omni
