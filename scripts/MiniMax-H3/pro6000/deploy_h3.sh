#!/bin/bash


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sglang serve \
  --model-path /mnt/SS4T/models/MiniMaxAI/MiniMax-H3 \
  --model-variant fl2va \
  --num-gpus 1 \
  --ulysses-degree 1 \
  --quantization fp8 \
  --performance-mode speed \
  --layerwise-offload-components text_encoder \
  --pin-cpu-memory false \
  --host 0.0.0.0 \
  --port 30052