#!/bin/bash
# Tier 1 experiment — vary ONLY Cache-DiT residual_diff_threshold vs tier0 (0.04 -> 0.20).
# Baseline: deploy_tier0.sh (DLO resident20 + Cache-DiT high/0.04).
#
# Hypothesis: raising the skip threshold lets more low-residual-diff DiT blocks
# be cached/skipped, cutting per-step compute. tier0's 0.04 is conservative
# (quality-first); 0.20 is the doc-suggested "more speed" point.
# TRADE-OFF: more skipping -> faster but lower fidelity. Re-run a quality gate
# (SSIM/PSNR vs a lossless tier0 run, or visual diff) — do NOT assume parity.
#
# Isolate this variable: resident layers and everything else identical to tier0.
# Measure: per-step hit/skip count from --enable-cache-dit-summary + quality.

export CUDA_VISIBLE_DEVICES=0,1

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800

PROFILER_FLAGS=""
if [ "${PROFILER:-0}" = "1" ]; then
    PROFILER_FLAGS="--enable-diffusion-pipeline-profiler"
fi

# shellcheck disable=SC2086
vllm serve /data/models/modelscope/MiniMax/MiniMax-H3/FL2VA \
  --omni --trust-remote-code \
  --host 0.0.0.0 --port 8000 \
  --num-gpus 2 \
  --tensor-parallel-size 2 \
  --text-encoder-tp-size 2 \
  --usp 1 --ring 1 \
  --vae-patch-parallel-size 2 \
  --vae-parallel-mode tile --vae-use-tiling \
  --enable-distributed-layerwise-offload --dlo-no-use-allgather \
  --dlo-resident-layers 20 \
  --enforce-eager \
  --cache-backend cache_dit \
  --cache-config '{"Fn_compute_blocks":1,"Bn_compute_blocks":0,"max_warmup_steps":4,"residual_diff_threshold":0.20,"max_continuous_cached_steps":1,"enable_taylorseer":false}' \
  --enable-cache-dit-summary \
  $PROFILER_FLAGS \
  --diffusion-attention-backend CUDNN_ATTN
