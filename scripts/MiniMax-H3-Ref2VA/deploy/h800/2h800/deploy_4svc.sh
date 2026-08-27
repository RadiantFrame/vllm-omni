#!/bin/bash
# Launch 4 independent MiniMax-H3 FL2VA services on an 8xH800 machine.
# Each service = deploy.sh config (TP2 + FP8 + Cache-DiT high) on 2 GPUs.
#
#   service 0 -> GPUs 0,1 -> port 9000 (master 29500)
#   service 1 -> GPUs 2,3 -> port 9001 (master 29501)
#   service 2 -> GPUs 4,5 -> port 9002 (master 29502)
#   service 3 -> GPUs 6,7 -> port 9003 (master 29503)
#
# Per-service performance == deploy.sh measured numbers (480p/832x480/5s:
# 26.5s steady; ~50 GiB model resident per card). 4x aggregate for throughput.
#
# All 4 start concurrently in the background; per-service logs go to
# deploy_4svc.svc<N>.gpu<..>.log in LOG_DIR. PIDs are written to
# deploy_4svc.pids for `kill $(cat ...)`.
#
# MASTER_ADDR/PORT are pinned per service (29500+i): without them the
# torch.distributed TCP store picks a RANDOM port and concurrent launches
# can race into EADDRINUSE (observed on the 1h800 8-way launcher; see
# ../1h800/deploy_8svc.sh). No host-RAM concern here: unlike the 1h800
# cpu-offload services (which pin the full ~130G model in RAM each), TP2
# keeps weights GPU-resident (~50 GiB/card); only concurrent checkpoint
# reads (~4 x 120G from disk) touch host memory during startup.

set -euo pipefail

MODEL=${MODEL:-/data/models/modelscope/MiniMax/MiniMax-H3/FL2VA}
LOG_DIR="${LOG_DIR:-./logs}"
PORT_BASE="${PORT_BASE:-9000}"
PID_FILE="$LOG_DIR/deploy_4svc.pids"

# GPU pairs, one per service.
PAIRS=("0,1" "2,3" "4,5" "6,7")

# Cache-DiT H3 "high" profile (same as deploy.sh).
CACHE_CONFIG='{"Fn_compute_blocks":1,"Bn_compute_blocks":0,"max_warmup_steps":4,"residual_diff_threshold":0.04,"max_continuous_cached_steps":1,"enable_taylorseer":false}'

PROFILER_FLAGS=""
if [ "${PROFILER:-0}" = "1" ]; then
    PROFILER_FLAGS="--enable-diffusion-pipeline-profiler"
fi

mkdir -p "$LOG_DIR"
: > "$PID_FILE"

echo "Launching ${#PAIRS[@]} services (2h800 deploy.sh config, 2 GPUs each)..."
for i in "${!PAIRS[@]}"; do
    gpus="${PAIRS[$i]}"
    port=$((PORT_BASE + i))
    master_port=$((29500 + i))
    log="$LOG_DIR/deploy_4svc.svc${i}.gpu${gpus}.log"
    echo "  service $i: GPUs=$gpus  port=$port  master_port=$master_port  log=$log"

    CUDA_VISIBLE_DEVICES="$gpus" \
    MASTER_ADDR=127.0.0.1 \
    MASTER_PORT="$master_port" \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
    nohup vllm serve "$MODEL" \
      --omni --trust-remote-code \
      --host 0.0.0.0 --port "$port" \
      --num-gpus 2 \
      --num-weight-load-threads 8 \
      --tensor-parallel-size 2 \
      --usp 1 --ring 1 \
      --text-encoder-tp-size 2 \
      --quantization fp8 \
      --cache-backend cache_dit \
      --cache-config "$CACHE_CONFIG" \
      --enable-cache-dit-summary \
      --diffusion-compile-granularity regional \
      --vae-patch-parallel-size 2 \
      --vae-parallel-mode tile --vae-use-tiling \
      --diffusion-attention-backend FLASH_ATTN \
      $PROFILER_FLAGS \
      > "$log" 2>&1 &

    echo $! >> "$PID_FILE"
done

echo ""
echo "Launched ${#PAIRS[@]} services. PIDs: $(tr '\n' ' ' < "$PID_FILE")"
echo ""
echo "Tail a log:   tail -f $LOG_DIR/deploy_4svc.svc0.gpu0,1.log"
echo "Health check: for p in 0 1 2 3; do curl -s http://localhost:\$((9000+p))/health; echo; done"
echo "Concurrent benchmark:"
echo "  NUM_SERVICES=4 PORT_BASE=9000 bash ../../generate/generate.sh"
echo "Stop all:     kill \$(cat $PID_FILE)"
