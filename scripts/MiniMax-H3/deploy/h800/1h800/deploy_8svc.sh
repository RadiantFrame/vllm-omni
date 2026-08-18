#!/bin/bash
# Launch 8 independent MiniMax-H3 FL2VA services on an 8xH800 machine.
# Each service = deploy.sh config (1-GPU cpu-offload + BF16 + Cache-DiT) per
# GPU -- the throughput counterpart of the single-service deploy.sh.
#
#   service 0 -> GPU 0 -> port 9000
#   service 1 -> GPU 1 -> port 9001
#   ...
#   service 7 -> GPU 7 -> port 9007
#
# All 8 start concurrently in the background; per-service logs go to
# deploy_8svc.svc<N>.gpu<N>.log in LOG_DIR. PIDs are written to
# deploy_8svc.pids for `kill $(cat ...)`.
#
# !!! HOST RAM WARNING !!!
# Model-level CPU offload keeps the FULL model (~120-135 GiB per service:
# encoder ~51.5G + DiT ~62G + VAEs) in host memory; the GPU holds only the
# active phase. 8 concurrent services therefore need ~8 x 130 ~= 1 TiB+ of
# system RAM (this host has ~2 TiB -- verify with `free -g` before launching
# and remember ~70G is already in use at idle). If RAM ever gets tight, drop
# services (GPUS array) rather than cards.
#
# Per-service throughput == deploy.sh measured numbers (480p/832x480/5s:
# 52.8s steady; 58.9 GiB GPU per card). 8x aggregate for throughput.

set -euo pipefail

MODEL=${MODEL:-/data/models/modelscope/MiniMax/MiniMax-H3/FL2VA}
LOG_DIR="${LOG_DIR:-./logs}"
PORT_BASE="${PORT_BASE:-9000}"
PID_FILE="$LOG_DIR/deploy_8svc.pids"

# One GPU per service, 8 services.
GPUS=(0 1 2 3 4 5 6 7)

# Cache-DiT H3 "high" profile (same as deploy.sh).
CACHE_CONFIG='{"Fn_compute_blocks":1,"Bn_compute_blocks":0,"max_warmup_steps":4,"residual_diff_threshold":0.04,"max_continuous_cached_steps":1,"enable_taylorseer":false}'

PROFILER_FLAGS=""
if [ "${PROFILER:-0}" = "1" ]; then
    PROFILER_FLAGS="--enable-diffusion-pipeline-profiler"
fi

mkdir -p "$LOG_DIR"
: > "$PID_FILE"

echo "Launching ${#GPUS[@]} services (1h800 deploy.sh config, 1 GPU each)..."
for i in "${!GPUS[@]}"; do
    gpu="${GPUS[$i]}"
    port=$((PORT_BASE + i))
    log="$LOG_DIR/deploy_8svc.svc${i}.gpu${gpu}.log"
    # Explicit per-service MASTER_ADDR/PORT: without them the torch.distributed
    # TCP store picks a RANDOM port, and 8 concurrent launches can race into
    # EADDRINUSE (observed: svc0 died with "port 43717 ... address already in
    # use" while svc1-7 came up fine). Pinning 29500+i removes the race.
    master_port=$((29500 + i))
    echo "  service $i: GPU=$gpu  port=$port  master_port=$master_port  log=$log"

    CUDA_VISIBLE_DEVICES="$gpu" \
    MASTER_ADDR=127.0.0.1 \
    MASTER_PORT="$master_port" \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
    VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT="${VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT:-300}" \
    nohup vllm serve "$MODEL" \
      --omni --trust-remote-code \
      --host 0.0.0.0 --port "$port" \
      --num-gpus 1 \
      --enable-cpu-offload \
      --cache-backend cache_dit \
      --cache-config "$CACHE_CONFIG" \
      --enable-cache-dit-summary \
      --diffusion-attention-backend FLASH_ATTN \
      $PROFILER_FLAGS \
      > "$log" 2>&1 &

    echo $! >> "$PID_FILE"
done

echo ""
echo "Launched ${#GPUS[@]} services. PIDs: $(tr '\n' ' ' < "$PID_FILE")"
echo ""
echo "Tail a log:   tail -f $LOG_DIR/deploy_8svc.svc0.gpu0.log"
echo "Health check: for p in \$(seq 0 7); do curl -s http://localhost:\$((9000+p))/health; echo; done"
echo "Stop all:     kill \$(cat $PID_FILE)"
