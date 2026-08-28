#!/bin/bash

# Profiler: PROFILER=1 bash deploy.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800

PORT=${PORT:-9000}
NUM_WEIGHT_LOAD_THREADS=${NUM_WEIGHT_LOAD_THREADS:-8}
MODEL=${MODEL:-/data/models/modelscope/MiniMax/MiniMax-H3/Ref2VA}
# auto: partition inferred from MODEL path (FL2VA→fl2va, Ref2VA→ref2va, root→combined)
TASK_TYPE=${TASK_TYPE:-auto}

PROFILER_FLAGS=""
if [ "${PROFILER:-0}" = "1" ]; then
    PROFILER_FLAGS="--enable-diffusion-pipeline-profiler"
fi

vllm serve  ${MODEL} \
  --omni \
  --task-type ${TASK_TYPE} \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port ${PORT} \
  --num-gpus 8 \
  --tensor-parallel-size 4 \
  --usp 2 \
  --ring 1 \
  --text-encoder-tp-size 8 \
  --vae-patch-parallel-size 8 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --enable-cpu-offload \
  --num-weight-load-threads ${NUM_WEIGHT_LOAD_THREADS} \
  --diffusion-compile-granularity regional \
  --diffusion-attention-backend SAGE_ATTN \
  --quantization fp8 \
  --cache-backend cache_dit \
  --cache-config '{"Fn_compute_blocks":1,"Bn_compute_blocks":0,"max_warmup_steps":4,"residual_diff_threshold":0.04,"max_continuous_cached_steps":1,"enable_taylorseer":false}' \
  --enable-cache-dit-summary \
  $PROFILER_FLAGS
