#!/bin/bash
# vLLM-Omni mirror of ../sglang/scripts/MiniMax-H3/h800/deploy_h3.sh.
# sglang shape: --tp-size 2 --ulysses-degree 2  =>  TP2 x Ulysses2 = 4 GPUs
#   (= plan "Candidate C": weight-shard 2-way + sequence-shard 2-way).
# Mapping to vLLM-Omni:
#   --tensor-parallel-size 2 --usp 2     <- tp2 + ulysses2
#   --text-encoder-tp-size 4             <- shard the Qwen3-VL encoder (4, not 2: TE-TP2 hits a cpu_group assert bug on 4 GPUs)
#   --quantization fp8                   <- online W8A8 of the DiT (BF16 on disk, dynamic activation); ~halves DiT weight mem
#   FL2VA partition subdir               <- sglang --model-variant fl2va
# NOTE: FP8 is incompatible with --enable-distributed-layerwise-offload (CUTLASS FP8 rejects the offload weight stride).

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
  --text-encoder-tp-size 4 \
  --vae-patch-parallel-size 4 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --diffusion-attention-backend FLASH_ATTN \
  --omni
