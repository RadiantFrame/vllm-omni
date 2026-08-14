#!/bin/bash
# Tier 1 experiment — vary ONLY --dlo-resident-layers vs tier0 (20 -> 32).
# Baseline: deploy_tier0.sh (DLO resident20 + Cache-DiT high/0.04).
#
# Hypothesis: 480p/5s run peaked at ~18 GiB/card (14 GiB headroom on 32 GiB).
# Each DiT block ~615 MB; raising resident 20->32 adds ~7.4 GiB -> ~25 GiB/card,
# cutting streamed blocks 30->18 (~40% less H2D per step) while staying within
# a safe margin for longer prompts / Ref2VA activation growth.
#
# Isolate this variable: everything else identical to tier0.
# Measure: denoise_step_latency_ms (vs tier0 ~20.7 s/step at 1344x768) and peak
# HBM via nvidia-smi.

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
  --dlo-resident-layers 32 \
  --enforce-eager \
  --cache-backend cache_dit \
  --cache-config '{"Fn_compute_blocks":1,"Bn_compute_blocks":0,"max_warmup_steps":4,"residual_diff_threshold":0.04,"max_continuous_cached_steps":1,"enable_taylorseer":false}' \
  --enable-cache-dit-summary \
  $PROFILER_FLAGS \
  --diffusion-attention-backend CUDNN_ATTN
