#!/bin/bash
# MiniMax-H3 FL2VA on 1x H800 (80 GiB, Hopper SM90), host RAM ~2 TiB.
# FP8 VARIANT of deploy.sh -- retry enabled by upstream #5910 (d1e230c9),
# "Enable MiniMax-H3 global FP8 with DLO".
#
# Historical context: FP8 + cpu-offload on one 80G card previously OOM'd at
# load time (logs/h3_0818_1h800.log, 2026-08-18: the per-layer online
# BF16->FP8 conversion accumulated on GPU alongside the resident encoder and
# died at 78.05G). #5910 adds _stream_online_quant_weights_to_cpu: each layer
# is offloaded to CPU AS SOON AS its quantization completes, eliminating the
# accumulation. This script re-tests FP8 on that fix.
#
# If it works, expected wins vs deploy.sh (BF16):
#   - DiT swaps halve (FP8 ~31G vs BF16 ~62G per direction over PCIe)
#   - FP8 GEMM speedup during denoise
#   - single-card steady 52.8s (BF16 measured) should drop meaningfully
#   - quality: officially qualified (LPIPS 0.116 / PSNR 23.6 / audio cosine 0.96)
# If it still OOMs at load: fall back to deploy.sh (BF16) and record the
#   failure -- 80G may still be short for the encoder-resident phase.
#
# Everything else identical to deploy.sh: cpu-offload (mutual exclusivity of
# encoder/DiT on the GPU), Cache-DiT "high", FLASH_ATTN = FA3 on Hopper,
# regional compile (no --enforce-eager). First request = compile warmup;
# request #2 may still be lazy-init settling; measure from #3.
# NOTE: per #5910 the default --quantization fp8 now covers BOTH the DiT and
# the Qwen3-VL text decoder (vision tower / VAEs / FP32 projections stay
# unquantized). Encoder residency therefore shrinks too (~51.5G BF16 ->
# ~26G FP8), further easing the 80G budget.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800
# Reduce allocator fragmentation: makes the ~44G headroom actually usable
# (suggested by PyTorch's own OOM diagnostics; zero-risk, esp. for 768p).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Post-denoise stages (DiT swap-out over PCIe + VAE decode + CPU MP4 encode)
# can exceed the engine's default 30s async-output timeout -> request aborted
# AFTER the video was actually generated.
export VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT=300

MODEL=${MODEL:-/data/models/modelscope/MiniMax/MiniMax-H3/FL2VA}
PORT=${PORT:-9000}

echo "Starting MiniMax-H3 FL2VA on 1xH800 (cpu-offload + FP8 + Cache-DiT high + FA3), port $PORT ..."

# shellcheck disable=SC2086
vllm serve "${MODEL}" \
  --omni --trust-remote-code \
  --host 0.0.0.0 --port "${PORT}" \
  --num-gpus 1 \
  --quantization fp8 \
  --cache-backend cache_dit \
  --cache-config '{"Fn_compute_blocks":1,"Bn_compute_blocks":0,"max_warmup_steps":4,"residual_diff_threshold":0.04,"max_continuous_cached_steps":1,"enable_taylorseer":false}' \
  --enable-cache-dit-summary \
  --diffusion-attention-backend FLASH_ATTN
