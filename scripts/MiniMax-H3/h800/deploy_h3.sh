#!/bin/bash
# vLLM-Omni mirror of ../sglang/scripts/MiniMax-H3/h800/deploy_h3.sh.
# sglang shape: --tp-size 2 --ulysses-degree 2  =>  TP2 x Ulysses2 = 4 GPUs
#   (= plan "Candidate C": weight-shard 2-way + sequence-shard 2-way), BF16.
# Mapping to vLLM-Omni:
#   --tensor-parallel-size 2 --usp 2     <- tp2 + ulysses2
#   --text-encoder-tp-size 2             <- shard the Qwen3-VL encoder (sglang TP2 does this implicitly)
#   FL2VA partition subdir               <- sglang --model-variant fl2va
# If BF16 OOMs on 80GB, append:  --quantization fp8

export CUDA_VISIBLE_DEVICES=4,5,6,7

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800

vllm serve /data/models/modelscope/MiniMax/MiniMax-H3/FL2VA \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --num-gpus 4 \
  --tensor-parallel-size 2 \
  --usp 2 \
  --text-encoder-tp-size 2 \
  --vae-patch-parallel-size 4 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --diffusion-attention-backend FLASH_ATTN \
  --omni
