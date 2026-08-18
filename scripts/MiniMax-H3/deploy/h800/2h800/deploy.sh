#!/bin/bash
# Tier 4 / 4b (2xH800) -- Cache-DiT (DBCache, H3 "high" profile) + TP2 + FP8.
# 2-GPU analog of the 4xH800 winner (deploy/h800/4h800/deploy_tier4_4b.sh =
# USP4 + TE-TP4 + FP8 + Cache-DiT = 114.77s steady-state).
#
# Why TP2 (not USP2) on 2 cards:
#   USP does NOT shard weights. USP2+FP8 would keep the FULL FP8 DiT (~33GB)
#   replicated per card + encoder shard (~25.7GB) ~= 65GB model resident ->
#   OOM risk at 1344x768 once compile workspace + activations pile on.
#   TP2 shards the DiT too (FP8 ~16.5GB/card -> ~48GB model resident, safe), and
#   head-sharding halves per-card attention just like seq-sharding would at sp=2.
#   This topology (TP2 + TE-TP2) is also what the repo's validated 2-GPU recipe uses.
#
# Stack: TP2 + text-encoder-TP2 (= world size, valid) + online FP8 + Cache-DiT
#        "high" (1.35x validated; residual_diff_threshold=0.04) + VAE pp2 tile +
#        FLASH_ATTN (FA3 on Hopper via fa3-fwd) + regional torch.compile
#        (no --enforce-eager). First request = compile warmup; don't count it.
#
# Cache tuning: for more speed raise residual_diff_threshold to 0.10-0.30 (re-run
#   quality gate); TeaCache (FL2VA-only) is the alternative backend -- mutually
#   exclusive with Cache-DiT. TaylorSeer stays OFF.
# Experimental: if you want to try USP2 anyway (--usp 2, drop --tensor-parallel-size),
#   watch for OOM at full 1344x768; drop resolution/duration first if it hits.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800

PORT=${PORT:-9000}
NUM_WEIGHT_LOAD_THREADS=${NUM_WEIGHT_LOAD_THREADS:-8}

PROFILER_FLAGS=""
if [ "${PROFILER:-0}" = "1" ]; then
    PROFILER_FLAGS="--enable-diffusion-pipeline-profiler"
fi

echo "Starting MiniMax-H3 FL2VA Tier 4 / 4b (TP2 + FP8 + Cache-DiT high) on 2xH800, port $PORT ..."

# shellcheck disable=SC2086
vllm serve /data/models/modelscope/MiniMax/MiniMax-H3/FL2VA \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port "$PORT" \
  --num-gpus 2 \
  --num-weight-load-threads "$NUM_WEIGHT_LOAD_THREADS" \
  --tensor-parallel-size 2 \
  --usp 1 \
  --ring 1 \
  --text-encoder-tp-size 2 \
  --quantization fp8 \
  --cache-backend cache_dit \
  --cache-config '{"Fn_compute_blocks":1,"Bn_compute_blocks":0,"max_warmup_steps":4,"residual_diff_threshold":0.04,"max_continuous_cached_steps":1,"enable_taylorseer":false}' \
  --enable-cache-dit-summary \
  --diffusion-compile-granularity regional \
  --vae-patch-parallel-size 2 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --diffusion-attention-backend FLASH_ATTN \
  $PROFILER_FLAGS \
  --omni
