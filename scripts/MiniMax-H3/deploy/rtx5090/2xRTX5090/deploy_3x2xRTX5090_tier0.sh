#!/bin/bash
# Launch 4 independent MiniMax-H3 FL2VA services on an 8xRTX5090 machine.
# Each service = deploy_tier0.sh config (DLO + Cache-DiT) on 2 GPUs.
#
#   service 0 -> GPUs 0,1 -> port 8000
#   service 1 -> GPUs 2,3 -> port 8001
#   service 2 -> GPUs 4,5 -> port 8002
#   service 3 -> GPUs 6,7 -> port 8003
#
# All 4 start concurrently in the background; per-service logs go to
# deploy_4x2xRTX5090_tier0.svc<N>.gpu<..>.log in LOG_DIR.
# PIDs are written to deploy_4x2xRTX5090_tier0.pids for `kill $(cat ...)`.

# !!! HOST RAM WARNING !!!
# DLO keeps rank-local weights in PINNED host memory. Each service holds ~one
# full FL2VA partition (~135 GiB) in RAM, so 4 concurrent services need
# ~4 x 135 = 540 GiB+ of system RAM (plus activations/overhead). Verify free
# RAM before launching (e.g. `free -g`); the 384 GiB figure in the recipe is
# per-single-server. If RAM is tight, reduce concurrency or lower
# --dlo-resident-layers does NOT help here (resident layers still retain a
# pinned CPU master copy).

set -euo pipefail

MODEL=/data/models/modelscope/MiniMax/MiniMax-H3/FL2VA
LOG_DIR="${LOG_DIR:-./logs}"
PORT_BASE="${PORT_BASE:-8000}"
PID_FILE="$LOG_DIR/deploy_4x2xRTX5090_tier0.pids"

# GPU id pairs, one per service.
PAIRS=("0,1" "2,3" "4,5")

# tier0 Cache-DiT config (H3 "high" profile).
CACHE_CONFIG='{"Fn_compute_blocks":1,"Bn_compute_blocks":0,"max_warmup_steps":4,"residual_diff_threshold":0.04,"max_continuous_cached_steps":1,"enable_taylorseer":false}'

PROFILER_FLAGS=""
if [ "${PROFILER:-0}" = "1" ]; then
    PROFILER_FLAGS="--enable-diffusion-pipeline-profiler"
fi

mkdir -p "$LOG_DIR"
: > "$PID_FILE"

echo "Launching ${#PAIRS[@]} services (tier0 config)..."
for i in "${!PAIRS[@]}"; do
    gpus="${PAIRS[$i]}"
    port=$((PORT_BASE + i))
    log="$LOG_DIR/deploy_4x2xRTX5090_tier0.svc${i}.gpu${gpus}.log"
    echo "  service $i: GPUs=$gpus  port=$port  log=$log"

    CUDA_VISIBLE_DEVICES="$gpus" \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
    nohup vllm serve "$MODEL" \
      --omni --trust-remote-code \
      --host 0.0.0.0 --port "$port" \
      --num-gpus 2 \
      --tensor-parallel-size 2 \
      --text-encoder-tp-size 2 \
      --usp 1 --ring 1 \
      --vae-patch-parallel-size 2 \
      --vae-parallel-mode tile --vae-use-tiling \
      --enable-distributed-layerwise-offload --dlo-no-use-allgather \
      --dlo-resident-layers 20 \
      --enforce-eager \
      --cache-backend cache_dit \
      --cache-config "$CACHE_CONFIG" \
      --enable-cache-dit-summary \
      $PROFILER_FLAGS \
      --diffusion-attention-backend CUDNN_ATTN \
      > "$log" 2>&1 &

    echo $! >> "$PID_FILE"
done

echo ""
echo "Launched ${#PAIRS[@]} services. PIDs: $(tr '\n' ' ' < "$PID_FILE")"
echo ""
echo "Tail a log:   tail -f $LOG_DIR/deploy_3x2xRTX5090_tier0.svc0.gpu0,1.log"
echo "Health check: for p in 0 1 2; do curl -s http://localhost:\$((8000+p))/health; echo; done"
echo "Stop all:     kill \$(cat $PID_FILE)"
