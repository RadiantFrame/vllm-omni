#!/bin/bash
# Tier 4 / 4a -- TeaCache (FL2VA) stacked on Candidate A (the latency winner).
# Ref: plan Tier 4 (4a). Builds on deploy_tier3_A.sh
#      (USP4 + TE-TP4 + FP8 = 155.74s steady-state).
#
# Cache: --cache-backend tea_cache, rel_l1_thresh=0.17 (H3 registered default).
#        TeaCache caches/skips similar denoise steps -> expected ~1.3-1.5x over A.
# FL2VA only: a Ref2VA-only server REJECTS TeaCache; a combined server runs Ref2VA
#        uncached. Your workload is FL2VA, so this is fine.
# Mutually exclusive with Cache-DiT (deploy_tier4_4b.sh) -- run ONE, not both.
# Quality gate: no published H3 TeaCache LPIPS/PSNR number -> compare same
#        seed/shape with vs without cache (LPIPS/PSNR + audio spectral cosine).

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800

PORT=${PORT:-8000}
NUM_WEIGHT_LOAD_THREADS=${NUM_WEIGHT_LOAD_THREADS:-8}

PROFILER_FLAGS=""
if [ "${PROFILER:-0}" = "1" ]; then
    PROFILER_FLAGS="--enable-diffusion-pipeline-profiler"
fi

echo "Starting MiniMax-H3 FL2VA Tier 4 / 4a (Candidate A + TeaCache 0.17) on port $PORT ..."

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
  --cache-backend tea_cache \
  --cache-config '{"rel_l1_thresh":0.17}' \
  --diffusion-compile-granularity regional \
  --vae-patch-parallel-size 4 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --diffusion-attention-backend FLASH_ATTN \
  $PROFILER_FLAGS \
  --omni
