#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=4500
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

vllm serve /data/models/modelscope/MiniMax/MiniMax-H3/FL2VA \
  --omni \
  --task-type fl2va \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 9000 \
  --num-gpus 8 \
  --tensor-parallel-size 4 \
  --usp 2 \
  --ring 1 \
  --text-encoder-tp-size 8 \
  --vae-patch-parallel-size 8 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --num-weight-load-threads 1 \
  --diffusion-compile-granularity regional \
  --diffusion-attention-backend SAGE_ATTN \
  --quantization fp8 \
  --enforce-eager \
  --lora-backend peft \
  --lora-path /data/models/modelscope/lightx2v/Minimax-h3-Turbo/minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors
