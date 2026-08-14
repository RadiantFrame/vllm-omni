#!/bin/bash
# Tier 0 / BASE deploy for MiniMax-H3 on 4xH800.
# Clean baseline to A/B optimizations against (FP8 / Cache-DiT / TeaCache ...).
# Ref: the 4xH800 H3 optimization plan, Tier 0 (baseline) + Tier 3 candidate D.
#
# Config = official 4xH100(Hopper) perf baseline (commit #5836,
# tests/dfx/perf/tests/test_minimax_h3_vllm_omni.json), adapted unchanged to H800:
#   --usp 4 --ring 1 --use-hsdp --hsdp-shard-size 4 --text-encoder-tp-size 4
#   BF16 (NO --quantization), NO cache backend, VAE pp4 tile, regional torch.compile
#   (no --enforce-eager), FLASH_ATTN (FA3 on Hopper).
# Peak ~54-55 GB/GPU; official H100 measured FL2VA 38.1s/8-step.
#
# A/B vs deploy.sh (Candidate C + FP8): run one server at a time on the same port,
# reuse generate.sh for the FL2VA request. Remember request #1 is the compile
# warmup — do NOT count it (Tier 0: warmup 1 + measure >=3).

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800

PORT=${PORT:-8000}

# Stage profiler (text_encoder / diffuse / vae.decode breakdown). Default OFF for
# clean latency baselines; PROFILER=1 to inspect per-stage timing.
PROFILER_FLAGS=""
if [ "${PROFILER:-0}" = "1" ]; then
    PROFILER_FLAGS="--enable-diffusion-pipeline-profiler"
fi

echo "Starting MiniMax-H3 FL2VA BASE (Tier 0 / Candidate D, BF16) on port $PORT ..."

# shellcheck disable=SC2086
vllm serve /data/models/modelscope/MiniMax/MiniMax-H3/FL2VA \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port "$PORT" \
  --num-gpus 4 \
  --usp 4 \
  --ring 1 \
  --use-hsdp \
  --hsdp-shard-size 4 \
  --text-encoder-tp-size 4 \
  --vae-patch-parallel-size 4 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --diffusion-attention-backend FLASH_ATTN \
  $PROFILER_FLAGS \
  --omni
