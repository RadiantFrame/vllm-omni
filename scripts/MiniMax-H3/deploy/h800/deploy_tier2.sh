#!/bin/bash
# Tier 2 deploy for MiniMax-H3 on 4xH800.
# Builds on deploy_tier1.sh by enabling online FP8 (plan Tier 2) -- the first
# tier with a MEASURABLE inference-latency / memory benefit.
#
# Fit-strategy swap vs Tier 1: Tier 0/1 fit 80GB in BF16 via HSDP4. Tier 2 instead
# fits via online FP8 (DiT linears -> W8A8, ~half DiT weight mem) and DROPS HSDP,
# because FP8+HSDP is unvalidated on H3 (the hsdp_fp8.py compat patch is not verified
# for H3). FP8 alone (USP4 + text-encoder-TP4) fits comfortably (~55 GB/GPU; this is
# the validated "Candidate A" the user already ran at 55.6 GiB/worker).
#
# Expected vs tier1 (BF16): lower per-step denoise latency (FP8 GEMM) + lower mem,
# at the cost of FP8 quantization. H3 FP8 is validated within the quality gate
# (LPIPS 0.116 / PSNR 23.6 dB / audio spectral cosine 0.96) -- run the gate if
# output fidelity matters.
#
# Incompat: FP8 is NOT compatible with --enable-distributed-layerwise-offload
# (CUTLASS FP8 rejects the offload weight stride). Confirm via startup log:
#   Selected CutlassFP8ScaledMMLinearKernel for Fp8PerTensorOnlineLinearMethod

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800

PORT=${PORT:-8000}
NUM_WEIGHT_LOAD_THREADS=${NUM_WEIGHT_LOAD_THREADS:-8}

PROFILER_FLAGS=""
if [ "${PROFILER:-0}" = "1" ]; then
    PROFILER_FLAGS="--enable-diffusion-pipeline-profiler"
fi

echo "Starting MiniMax-H3 FL2VA Tier 2 (Candidate A: USP4 + TE-TP4 + FP8) on port $PORT ..."

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
