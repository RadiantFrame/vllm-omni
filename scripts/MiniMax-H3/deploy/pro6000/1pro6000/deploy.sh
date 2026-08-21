#!/bin/bash
# MiniMax-H3 FL2VA on 1x RTX PRO 6000 Blackwell (96 GiB, SM120), host RAM 125 GiB.
#
# FP8 VARIANT of deploy.sh — the preferred single-card config when it works.
#
# Key idea: online FP8 halves the DiT from 62 G to 31 G, which halves every
# host-RAM pressure point of model-level CPU offload:
#   - static host copy (encoder):          48 G (unchanged)
#   - transition peak (both copies):       48 + 31 = 79 G  (BF16: 110 G -> froze)
#   - GPU during denoise:                  31 G + activations (headroom for 720p+)
#   - PCIe swap traffic:                   31 G per direction (half)
# plus FP8 GEMM speedup on Blackwell. Officially quality-qualified for H3
# (LPIPS 0.116 / PSNR 23.6 / audio cosine 0.96).
#
# Compatibility notes:
#   - FP8 is documented as incompatible with LAYERWISE (DLO) offload (weight
#     stride); there is NO such gate for model-level (sequential) offload, and
#     --enable-cpu-offload forces load_device="cpu" so there is no GPU-side
#     BF16->FP8 conversion transient either. If this combination turns out to
#     hit an untested path, fall back to deploy.sh (BF16, less headroom).
#   - With ~91 G peaks no extra swap machinery is needed; keep
#     VLLM_OMNI_PIN_CPU_MEMORY=0 only as cheap insurance.
#
# mmap+DLO (the zero-RSS alternative) is NOT currently available: the mmap path
# only activates for DLO+AllGather with dp>1, and H3 sets
# _supports_mmap_loading=False (grouped-QKV/fused-MLP layout adapters not yet
# verified) — see pipeline_minimax_h3.py and distributed_layerwise_backend.py.
#
# Run inside tmux (not a VS Code terminal) so a desktop crash cannot take the
# server down; see deploy.sh for the full story.

export CUDA_VISIBLE_DEVICES=0

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800
export VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT=300

MODEL=/mnt/SS4T/models/MiniMaxAI/MiniMax-H3/FL2VA
# Port 8000 is taken by the VS Code port-forwarding service on this host.
# Client must override: BASE_URL=http://localhost:9000 bash ../../generate/generate_480p_5s_nsvc.sh
PORT=9000

# --- systemd-oomd must be STOPPED on this host before serving ---
# Ubuntu's systemd-oomd kills whole cgroup scopes based on PSI memory-pressure
# stalls (not actual exhaustion). Any swap-heavy workload — and CPU-offload
# transitions are exactly that — gets scopes (the server, even the desktop)
# massacred mid-request. `systemctl disable` only affects boot; the running
# instance must be stopped explicitly (this host already ran `disable`, so the
# stop also persists across reboots):
#   sudo systemctl stop systemd-oomd.socket systemd-oomd.service   # stop now
#   systemctl is-active systemd-oomd systemd-oomd.socket           # both must say inactive
#   journalctl --user | grep "systemd-oomd killed"                 # post-mortem: it signs its kills
# To restore the distro default later:
#   sudo systemctl enable --now systemd-oomd.socket systemd-oomd.service
# With oomd gone, the MemoryMax fuse below is the targeted backstop instead.
#
# Fuse: observed steady-state cgroup usage is ~101 G (worker's two component
# copies + staging), plus page cache charged to the scope. 105 G leaves ~4 G
# of legitimate headroom (bigger shapes / Ref2VA) while still stopping a
# runaway well below the ~120 G system-freeze territory. Check
# memory.events oom_kill if the server ever dies here before raising it.
# systemd-run --user --scope --quiet -p MemoryMax=105G \
vllm serve "${MODEL}" \
  --omni --trust-remote-code \
  --host 0.0.0.0 --port "${PORT}" \
  --num-gpus 1 \
  --quantization fp8 \
  --diffusion-compile-granularity regional \
  --num-weight-load-threads 8 \
  --cache-backend cache_dit \
  --cache-config '{"Fn_compute_blocks":1,"Bn_compute_blocks":0,"max_warmup_steps":4,"residual_diff_threshold":0.04,"max_continuous_cached_steps":1,"enable_taylorseer":false}' \
  --enable-cache-dit-summary \
  --diffusion-attention-backend CUDNN_ATTN
