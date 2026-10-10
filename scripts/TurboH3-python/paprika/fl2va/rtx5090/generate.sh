#!/usr/bin/env bash
curl -sS -X POST http://localhost:9000/v1/videos/sync \
  -F "prompt=${PROMPT}" \
  -F fps=24 \
  -F num_inference_steps=8 \
  -F seed=0 \
  -F short_edge=768 \
  -F aspect_ratio=adaptive \
  -F flow_shift=6 \
  -F 'extra_params={"task": "fl2va", "duration": 5, "audio_flow_shift": 3.0}' \
  -F 'lora={"name": "h3-turbo-fl2v-v1.0-768p", "path": "/data/models/modelscope/lightx2v/Minimax-h3-Turbo/minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors", "scale": 1.0}' \
  -F 'input_reference=@/data/jw/workspace/vllm-omni/inputs/i2va/references/4a3a90bf9100_KDmcbkhzYo5sjjxr9FqcVmWVnzb.png;type=image/png'
