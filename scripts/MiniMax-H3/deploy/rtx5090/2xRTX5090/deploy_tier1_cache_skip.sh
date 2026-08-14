#!/bin/bash
# Tier 1 experiment — add step-level skip on top of tier0's block caching.
# Baseline: deploy_tier0.sh (DLO resident20 + Cache-DiT high/0.04).
#
# Hypothesis: tier0 caches individual DiT blocks; adding
# "scm_steps_mask_policy":"medium" enables whole-step skipping (the doc's
# "aggressive step-skip" knob), stacking a second layer of compute reduction.
# This is the most aggressive cache profile in the tier1 set — biggest expected
# speedup AND biggest quality risk.
# TRADE-OFF: re-run a quality gate (SSIM/PSNR or visual diff vs tier0) before
# trusting output.
#
# Isolate this variable: resident layers and threshold identical to tier0;
# only the steps_mask_policy field is added.

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
  --cache-config '{"Fn_compute_blocks":1,"Bn_compute_blocks":0,"max_warmup_steps":4,"residual_diff_threshold":0.04,"max_continuous_cached_steps":1,"scm_steps_mask_policy":"medium","enable_taylorseer":false}' \
  --enable-cache-dit-summary \
  $PROFILER_FLAGS \
  --diffusion-attention-backend CUDNN_ATTN
