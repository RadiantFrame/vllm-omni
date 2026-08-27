#!/bin/bash
# Launch 2 independent MiniMax-H3 FL2VA services on an 8xH800 machine.
# Each service = deploy_tier4_4b.sh config (USP4 + TE-TP4 + FP8 + Cache-DiT
# high) on 4 GPUs -- the lowest-latency layout's throughput counterpart.
# Naming follows the deploy_tier0_4svc.sh convention (tier config + Nsvc).
#
#   service 0 -> GPUs 0,1,2,3 -> port 9000 (master 29500)
#   service 1 -> GPUs 4,5,6,7 -> port 9001 (master 29501)
#
# Per-service performance == deploy_tier4_4b.sh measured numbers
# (768p/1344x768/8s: 114.77s steady; 480p class ~2x faster than 2-GPU TP2).
# 2x aggregate for throughput.
#
# All 2 start concurrently in the background; per-service logs go to
# deploy_tier4_4b_2svc.svc<N>.gpu<..>.log in LOG_DIR. PIDs are written to
# deploy_tier4_4b_2svc.pids for `kill $(cat ...)`.
#
# MASTER_ADDR/PORT are pinned per service (29500+i): without them the
# torch.distributed TCP store picks a RANDOM port and concurrent launches
# can race into EADDRINUSE (observed on the 1h800 8-way launcher). No
# host-RAM concern: USP4+FP8 keeps weights GPU-resident (~55 GiB/card);
# only concurrent checkpoint reads (~2 x 120G) touch host memory at startup.
#
# Layout trade-off on this machine (see sibling READMEs):
#   2x4-GPU (this): lowest per-request latency of the multi-service options;
#                   best when latency matters more than aggregate throughput.
#   4x2-GPU (../2h800/deploy_4svc.sh): best aggregate throughput (measured
#                   8.81 videos/min @480p, 97% linear, only +2.8% contention).
#   8x1-GPU (../1h800/deploy_8svc.sh): only when one-card-per-service is a
#                   hard constraint (cpu-offload, +16.5% contention).

set -euo pipefail

MODEL=${MODEL:-/data/models/modelscope/MiniMax/MiniMax-H3/FL2VA}
LOG_DIR="${LOG_DIR:-./logs}"
PORT_BASE="${PORT_BASE:-9000}"
PID_FILE="$LOG_DIR/deploy_tier4_4b_2svc.pids"

# GPU quads, one per service.
QUADS=("0,1,2,3" "4,5,6,7")

# Cache-DiT H3 "high" profile (same as deploy_tier4_4b.sh).
CACHE_CONFIG='{"Fn_compute_blocks":1,"Bn_compute_blocks":0,"max_warmup_steps":4,"residual_diff_threshold":0.04,"max_continuous_cached_steps":1,"enable_taylorseer":false}'

PROFILER_FLAGS=""
if [ "${PROFILER:-0}" = "1" ]; then
    PROFILER_FLAGS="--enable-diffusion-pipeline-profiler"
fi

mkdir -p "$LOG_DIR"
: > "$PID_FILE"

echo "Launching ${#QUADS[@]} services (4h800 tier4_4b config, 4 GPUs each)..."
for i in "${!QUADS[@]}"; do
    gpus="${QUADS[$i]}"
    port=$((PORT_BASE + i))
    master_port=$((29500 + i))
    log="$LOG_DIR/deploy_tier4_4b_2svc.svc${i}.gpu${gpus}.log"
    echo "  service $i: GPUs=$gpus  port=$port  master_port=$master_port  log=$log"

    CUDA_VISIBLE_DEVICES="$gpus" \
    MASTER_ADDR=127.0.0.1 \
    MASTER_PORT="$master_port" \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
    nohup vllm serve "$MODEL" \
      --omni --trust-remote-code \
      --host 0.0.0.0 --port "$port" \
      --num-gpus 4 \
      --num-weight-load-threads 8 \
      --usp 4 --ring 1 \
      --text-encoder-tp-size 4 \
      --quantization fp8 \
      --cache-backend cache_dit \
      --cache-config "$CACHE_CONFIG" \
      --enable-cache-dit-summary \
      --diffusion-compile-granularity regional \
      --vae-patch-parallel-size 4 \
      --vae-parallel-mode tile --vae-use-tiling \
      --diffusion-attention-backend FLASH_ATTN \
      $PROFILER_FLAGS \
      > "$log" 2>&1 &

    echo $! >> "$PID_FILE"
done

echo ""
echo "Launched ${#QUADS[@]} services. PIDs: $(tr '\n' ' ' < "$PID_FILE")"
echo ""
echo "Tail a log:   tail -f $LOG_DIR/deploy_tier4_4b_2svc.svc0.gpu0,1,2,3.log"
echo "Health check: for p in 0 1; do curl -s http://localhost:\$((9000+p))/health; echo; done"
echo "Concurrent benchmark:"
echo "  NUM_SERVICES=2 PORT_BASE=9000 bash ../../generate/generate.sh 2>/dev/null ||"
echo "  NUM_SERVICES=2 PORT_BASE=9000 bash ../../generate/generate.sh"
echo "Stop all:     kill \$(cat $PID_FILE)"
