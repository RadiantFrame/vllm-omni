#!/bin/bash
# MiniMax-H3 on two RTX 5090s (32 GiB HBM each).
# Memory-first serving configuration for recipes/MiniMaxAI/MiniMax-H3-5090.md.
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
# FL2VA and Ref2VA are separate 135 GiB partitions; run one server at a time.
# For Ref2VA, stop this server and restart with the Ref2VA partition path.

export CUDA_VISIBLE_DEVICES=0,1

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800

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
  --diffusion-attention-backend CUDNN_ATTN
