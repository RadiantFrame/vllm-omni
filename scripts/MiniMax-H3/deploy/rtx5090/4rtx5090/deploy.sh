#!/bin/bash
# MiniMax-H3 FL2VA on 4x RTX 5090 (32 GiB HBM each), single service.
# SPEED VARIANT (A/B against deploy.sh): TP4 + online FP8 + NO DLO +
# regional torch.compile + cuda graph + Cache-DiT.
#
# Potentially the fastest 4x5090 config (no H2D streaming + cuda graph +
# FP8 ~12% + Cache-DiT), but UNVALIDATED on 5090 — run the validation gates
# below before trusting it. If any gate fails, fall back to deploy.sh.
#
# Diff vs deploy.sh (the safe primary):
#   - DROPS: --enable-distributed-layerwise-offload, --dlo-no-use-allgather,
#           --dlo-resident-layers, --enforce-eager
#   - ADDS:  --quantization fp8  (DiT W8A8 online; encoder/VAE stay BF16)
#   - Dropping --enforce-eager enables the default regional torch.compile +
#     cuda graph (the main speedup source). FP8 is incompatible with DLO, so
#     the two configs are mutually exclusive.
#
# !!! STATUS (2026-08-14, RTX 5090 32 GiB): CONFIRMED OOM — DO NOT USE on 4x5090 !!!
# Online FP8 loads BF16 weights to GPU then converts to FP8 per layer, so BF16
# original + FP8 output coexist transiently. At TP4 the BF16 model alone is
# ~38.8 GiB/card > 31.4 GiB -> OOM in fp8.py:159 scaled_fp8_quant during
# process_weights_after_loading. The FP8 kernel itself IS fine on 5090
# (CutlassFP8ScaledMMLinearKernel + DeepGEMM init OK); this is purely a
# capacity limit. Not fixable by tuning. Use deploy.sh (DLO+BF16) instead.
# This script is kept for hosts with >=80 GiB cards. See README.md.
# The only way to get FP8 speed on 5090: an OFFLINE pre-quantized FP8 DiT
# checkpoint + DLO (no load-time transient, compatible with DLO). Not yet built.
#
# Profiler: PROFILER=1 bash deploy_fp8.sh

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Anti-fragmentation allocator: 768p/15s OOMed with 3.7G reserved-but-unallocated
# (1.5G alloc failed with 1.2G free + fragmentation); expandable segments fix that.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800

PROFILER_FLAGS=""
if [ "${PROFILER:-0}" = "1" ]; then
    PROFILER_FLAGS="--enable-diffusion-pipeline-profiler"
fi

# shellcheck disable=SC2086
vllm serve /data/models/modelscope/MiniMax/MiniMax-H3/FL2VA \
  --omni --trust-remote-code \
  --host 0.0.0.0 --port 9000 \
  --num-gpus 4 \
  --tensor-parallel-size 4 \
  --text-encoder-tp-size 4 \
  --usp 1 --ring 1 \
  --vae-patch-parallel-size 4 \
  --vae-parallel-mode tile --vae-use-tiling \
  --quantization fp8 \
  --enable-cpu-offload \
  --num-weight-load-threads 8 \
  --cache-backend cache_dit \
  --cache-config '{"Fn_compute_blocks":1,"Bn_compute_blocks":0,"max_warmup_steps":4,"residual_diff_threshold":0.04,"max_continuous_cached_steps":1,"enable_taylorseer":false}' \
  --enable-cache-dit-summary \
  --diffusion-compile-granularity regional \
  $PROFILER_FLAGS \
  --diffusion-attention-backend SAGE_ATTN
