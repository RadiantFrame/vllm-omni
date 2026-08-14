#!/bin/bash
# MiniMax-H3 on two RTX 5090s (32 GiB HBM each).
# Memory-first serving configuration for recipes/MiniMaxAI/MiniMax-H3-5090.md,
# stacked with Cache-DiT to cut per-step DiT compute.
# Shape: 1344x768, 5 seconds.
#
# Topology: TP2 + Ulysses1/ring1 + VAE patch-parallel2.
# Distributed layerwise offload (DLO) keeps rank-local DiT weights in pinned host
# memory and streams them H2D per block (no AllGather). 20 resident DiT layers
# stay on HBM; the two-rank B300 capacity run for this shape peaked at ~27.7 GiB
# per rank (50 steps) — re-measure peak HBM before raising --dlo-resident-layers.
# FP8/online quantization is NOT used: it is incompatible with
# --enable-distributed-layerwise-offload.
#
# Cache-DiT (DBCache, H3 "high" profile) skips recomputing low-residual-diff
# DiT blocks. The DLO backend handles skipped blocks via its _prev_hook sync
# prefetch fallback (distributed_layerwise_backend.py), so cache_dit + DLO is
# code-compatible. NOTE: no shipped script validates this exact combination on
# FL2VA — verify output quality after a run.
#   residual_diff_threshold=0.04  conservative (quality-first); raise to 0.20-0.30
#                                 for more speed after re-checking quality.
#   max_warmup_steps=4            steps always computed fully before caching kicks in.
#   max_continuous_cached_steps=1 cap on consecutive skipped steps.
# --enable-cache-dit-summary logs per-step hit/skip so the speedup is visible.
# Mutually exclusive with TeaCache — do not also pass a teacache cache-backend.
#
# FL2VA and Ref2VA are separate 135 GiB partitions; run one server at a time.
# For Ref2VA, stop this server and restart with the Ref2VA partition path.

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
  --cache-config '{"Fn_compute_blocks":1,"Bn_compute_blocks":0,"max_warmup_steps":4,"residual_diff_threshold":0.04,"max_continuous_cached_steps":1,"enable_taylorseer":false}' \
  --enable-cache-dit-summary \
  $PROFILER_FLAGS \
  --diffusion-attention-backend CUDNN_ATTN
