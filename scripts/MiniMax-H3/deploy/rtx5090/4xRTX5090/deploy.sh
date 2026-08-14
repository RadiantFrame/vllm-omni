#!/bin/bash
# MiniMax-H3 FL2VA on 4x RTX 5090 (32 GiB HBM each), single service.
# PRIMARY / SAFE config: TP4 + DLO(no-AllGather) + BF16 + Cache-DiT.
#
# Why this shape (no validated 4x5090 topology exists in-repo; ported from the
# constraints verified for 2x5090 + the 4-GPU H800 candidates):
#   - TP4 shards DiT weights 4-way (~15.5 GiB/rank) — the only TP degree that
#     lets BF16 fit on 32 GiB cards.
#   - text-encoder-tp-size 4: the Qwen3-VL encoder (~51.5 GiB BF16) is the
#     memory hotspot and defaults onto rank 0; TE-TP4 shards it 4-way
#     (64 attn heads / 8 KV heads both divisible by 4).
#   - usp 1 / ring 1: TP4 already fills the DiT group (4); 480p activations are
#     small, no need for sequence parallelism. (--usp 4 --tp 1 is an anti-pattern:
#     it replicates the full DiT on every card.)
#   - vae-patch-parallel-size 4 (must equal the DiT group) + native tile mode.
#   - DLO no-AllGather is the only BF16-compatible memory path; it also stages
#     the encoder/VAE on demand. resident_layers requires --dlo-no-use-allgather.
#   - enforce-eager: DLO streaming hooks break cuda-graph capture (mandatory).
#
# Tuning:
#   - resident_layers: at TP4 each resident block is ~half the TP2 size, so the
#     20 default is very safe. 480p is compute-bound (raising it gave no latency
#     gain in earlier tests), so keep 20 unless profiling shows H2D overhead.
#   - Real speed levers on this path: Cache-DiT residual_diff_threshold
#     (0.04 -> 0.20 measured ~16% faster, re-check quality), or switch to the
#     FP8 variant (deploy_fp8.sh) for regional-compile + cuda-graph + FP8.
#
# Profiler: PROFILER=1 bash deploy.sh  (enables --enable-diffusion-pipeline-profiler)

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

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
  --num-gpus 4 \
  --tensor-parallel-size 4 \
  --text-encoder-tp-size 4 \
  --usp 1 --ring 1 \
  --vae-patch-parallel-size 4 \
  --vae-parallel-mode tile --vae-use-tiling \
  --enable-distributed-layerwise-offload --dlo-no-use-allgather \
  --dlo-resident-layers 20 \
  --enforce-eager \
  --num-weight-load-threads 8 \
  --cache-backend cache_dit \
  --cache-config '{"Fn_compute_blocks":1,"Bn_compute_blocks":0,"max_warmup_steps":4,"residual_diff_threshold":0.04,"max_continuous_cached_steps":1,"enable_taylorseer":false}' \
  --enable-cache-dit-summary \
  $PROFILER_FLAGS \
  --diffusion-attention-backend CUDNN_ATTN
