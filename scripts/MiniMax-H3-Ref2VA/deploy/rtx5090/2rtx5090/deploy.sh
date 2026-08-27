#!/bin/bash
# MiniMax-H3 FL2VA on two RTX 5090s (32 GiB HBM each), single service.
# BEST-PERFORMANCE config after the FP8+DLO enablement landed in main
# (#6279 online FP8 with DLO; text encoder quantization; quantized weights
# streamed layer-wise to CPU by the loader).
#
# What changed vs the old BF16 profile:
#   - --quantization fp8 is GLOBAL: quantizes BOTH the DiT and the Qwen3-VL
#     text encoder (online W8A8; encoder supports online FP8 only).
#   - Offload strategy switched DLO -> MODEL-LEVEL (--enable-cpu-offload):
#     with FP8 the whole DiT fits on HBM (15.5 GiB/card), so per-block
#     streaming is unnecessary — the DiT stays fully resident through all
#     denoise steps (zero per-step H2D) and encoder/VAE swap in only during
#     their stages (mutual exclusion). Loader handles online-quant + CPU
#     offload via offload_after_quant (quantize on device, then offload).
#   - Regional torch.compile + cuda graph enabled (enforce-eager REMOVED):
#     model-level swap hooks only fire at stage boundaries, per-block compile
#     inside the DiT should be unaffected. NOTE: the FIRST request pays the
#     compile warmup (~30s+) — exclude it from latency measurements. If graph
#     capture crashes at startup or on request 1, add --enforce-eager back.
#   - Cache-DiT at residual_diff_threshold=0.04 (official "high" profile,
#     validated SSIM 0.9709 / PSNR 34.98). Default policy: always use the
#     official R=0.04 unless explicitly stated otherwise. R=0.20 measured
#     -16% latency on this host but its quality gate was never run.
#   - Free perf from main (automatic): fused H3 modulation w/ FP32 accum
#     (#6281), fused SwiGLU (#6283), configurable _ASYNC_OUTPUT_TIMEOUT.
#
# NOTE: recipes/MiniMaxAI/MiniMax-H3-5090.md predates FP8+DLO — this config
# is ahead of the recipe. Verify with a quality gate before production.
#
# Profiler: PROFILER=1 bash deploy.sh

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
  --host 0.0.0.0 --port 9000 \
  --num-gpus 2 \
  --tensor-parallel-size 2 \
  --text-encoder-tp-size 2 \
  --usp 1 --ring 1 \
  --vae-patch-parallel-size 2 \
  --vae-parallel-mode tile --vae-use-tiling \
  --quantization fp8 \
  --enable-cpu-offload \
  --diffusion-compile-granularity regional \
  --num-weight-load-threads 8 \
  --cache-backend cache_dit \
  --cache-config '{"Fn_compute_blocks":1,"Bn_compute_blocks":0,"max_warmup_steps":4,"residual_diff_threshold":0.04,"max_continuous_cached_steps":1,"enable_taylorseer":false}' \
  --enable-cache-dit-summary \
  $PROFILER_FLAGS \
  --diffusion-attention-backend CUDNN_ATTN
