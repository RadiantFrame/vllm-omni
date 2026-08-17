#!/bin/bash
# MiniMax-H3 FL2VA on 1x RTX PRO 6000 Blackwell (96 GiB, SM120), host RAM 125 GiB.
#
# Topology: model-level CPU offload (SequentialOffloadHook) — the DiT (~62 G) and
# the Qwen3-VL encoder (~63 G) are mutually exclusive on the GPU; the inactive
# side sits pinned in host memory (~73 G peak < ~94 G available). Single GPU, so
# no TP/USP; VAE patch-parallel stays 1 (tiling is on by default).
#
# Why not the alternatives on THIS host:
#   - No offload: BF16 partition ~124 G > 96 GiB -> OOM (recipe: "TP1 is not an
#     option on this card").
#   - online FP8: incompatible with H3 layerwise offload; without offload the
#     encoder alone (63 G BF16) + FP8 DiT + load-time BF16/FP8 transient > 96 GiB.
#   - DLO: pins the full ~135 G partition in host RAM (recipe min 200 GiB) — this
#     host has 125 GiB. Also forces --enforce-eager and excludes FP8.
#   - sglang's "FP8 DiT resident + encoder-only offload" shape has no vLLM-Omni
#     equivalent (no component-scoped offload CLI).
#
# Cache-DiT R=0.04 (H3 "high" preset): official H200 1.35x; measured -26% E2E on
# 4xH800 (155.74 -> 114.77 s). Compatible with the sequential offload backend.
# Raise residual_diff_threshold to 0.10-0.20 for more speed ONLY after passing
# the quality gate (LPIPS/PSNR + audio spectral cosine, fixed prompt/seed).
#
# Attention: SM120 has no TRTLLM_ATTN (datacenter Blackwell only); CUDNN_ATTN is
# the PRO 6000 recipe's pinned choice. FLASH_ATTN would resolve to the
# experimental FA4 kernel (requires the [fa4] extra) — not the default here.
#
# No --enforce-eager: model-level offload keeps the default regional
# torch.compile; the first request pays ~30 s of compile warmup (exclude it
# from latency measurements).
#
# Practical notes:
#   - Host RAM is tight (~90-95 G pinned+overhead vs ~94 G available): stop other
#     large processes first. Pinned memory cannot swap (the 63 G swap only
#     cushions unpinned allocations).
#   - Start at 480p (832x480, 5 s): 2x96 G TP2 runs 5.57 s/step at 1344x768, so
#     expect ~2x that single-card. Client: ../../generate/generate_480p_5s.sh
#   - Ref2VA is a separate 135 G partition: stop this server and restart with
#     MODEL=.../MiniMax-H3/Ref2VA (never both at once on this host).
#   - Log anchors to verify: "Cache-dit enabled successfully" + cache summary,
#     "Resolved diffusion attention backend 'CUDNN_ATTN'" (beware "Falling back").

export CUDA_VISIBLE_DEVICES=0

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800
# Post-denoise stages (DiT swap-out over PCIe + VAE decode + CPU MP4 encode) can
# exceed the engine's default 30 s async-output timeout -> request aborted with
# TimeoutError AFTER the video was actually generated (see 2rtx5090 README).
export VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT=300
# Host RAM (125 GiB) barely fits the offloaded side (~62-63 G) plus process
# overhead: pinned copies cannot swap, and the transient pageable+pinned double
# copy during a swap OOM-killed the desktop IDE on request 2. Unpinned copies
# are swappable (63 G swap available) at some H2D transfer speed cost.
export VLLM_OMNI_PIN_CPU_MEMORY=0

MODEL=/mnt/SS4T/models/MiniMaxAI/MiniMax-H3/FL2VA
# Port 8000 is taken by the VS Code port-forwarding service on this host.
# Client must override: BASE_URL=http://localhost:9000 bash ../../generate/generate_480p_5s.sh
PORT=9000

# Fuse: a cgroup MemoryMax makes the worst case "server OOM-killed inside its
# own scope" instead of "whole machine swap-livelocked". systemd-oomd is
# disabled on this host (it killed whole desktop scopes); this per-service
# limit replaces it as the targeted backstop. Measured transition peak is
# ~109-110 GiB (idle RSS ~60 + D2H 61.7); 116 G sits above the legitimate
# peak but below the ~120 G freeze territory. Do NOT set this to
# ~108 G: the kill fires 1-2 GiB before phase 2 can engage (verified the hard
# way). Drop the systemd-run prefix if it is unavailable.
systemd-run --user --scope --quiet -p MemoryMax=116G \
vllm serve "${MODEL}" \
  --omni --trust-remote-code \
  --host 0.0.0.0 --port "${PORT}" \
  --num-gpus 1 \
  --enable-cpu-offload \
  --cache-backend cache_dit \
  --cache-config '{"Fn_compute_blocks":1,"Bn_compute_blocks":0,"max_warmup_steps":4,"residual_diff_threshold":0.04,"max_continuous_cached_steps":1,"enable_taylorseer":false}' \
  --enable-cache-dit-summary \
  --diffusion-attention-backend CUDNN_ATTN
