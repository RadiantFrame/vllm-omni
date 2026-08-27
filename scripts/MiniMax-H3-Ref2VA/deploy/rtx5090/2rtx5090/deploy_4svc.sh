#!/bin/bash
# Launch 4 independent MiniMax-H3 FL2VA services on an 8xRTX 5090 machine.
# Each service = deploy.sh config (TP2 + FP8 + model-level cpu-offload +
# regional compile + Cache-DiT 0.20) on 2 GPUs.
#
#   service 0 -> GPUs 0,1 -> port 9000 (master 29500)
#   service 1 -> GPUs 2,3 -> port 9001 (master 29501)
#   service 2 -> GPUs 4,5 -> port 9002 (master 29502)
#   service 3 -> GPUs 6,7 -> port 9003 (master 29503)
#
# !!! HOST RAM IS THE GATE HERE (unlike h800/2h800 whose weights stay on GPU):
# each service pins the FP8 model in host RAM. Post-FP8 estimate ~120 GiB per
# service (weights ~67G + 2 worker overheads ~52G) => 4 services ~480 GiB vs
# 503 GiB total — TIGHT. Check `free -g` before launching; if OOM-killed
# (worker exit code -9 during load), drop to deploy_2svc/deploy_3svc. BF16
# era this was impossible (4 x ~187G); FP8 is what makes 4-way borderline.
#
# MASTER_ADDR/PORT are pinned per service (29500+i): concurrent launches
# otherwise race on the torch.distributed TCP store's random port
# (EADDRINUSE; see ../../h800/1h800/deploy_8svc.sh).
# VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT=120: 4-way concurrent MP4 software encode
# under CPU contention exceeds the default 30s output wait (requests that
# actually completed would abort with HTTP 500; fixed via env in main).
#
# Per-service perf == deploy.sh (first request per service pays the regional
# compile warmup, ~30s+ — exclude from measurements).
# Logs: deploy_4svc.svc<N>.gpu<..>.log in LOG_DIR; PIDs in deploy_4svc.pids.

set -euo pipefail

MODEL=${MODEL:-/data/models/modelscope/MiniMax/MiniMax-H3/FL2VA}
LOG_DIR="${LOG_DIR:-./logs}"
PORT_BASE="${PORT_BASE:-9000}"
PID_FILE="$LOG_DIR/deploy_4svc.pids"

# GPU pairs, one per service.
PAIRS=("0,1" "2,3" "4,5" "6,7")

# Same as deploy.sh: Cache-DiT official "high" profile (R=0.04, validated).
CACHE_CONFIG='{"Fn_compute_blocks":1,"Bn_compute_blocks":0,"max_warmup_steps":4,"residual_diff_threshold":0.04,"max_continuous_cached_steps":1,"enable_taylorseer":false}'

PROFILER_FLAGS=""
if [ "${PROFILER:-0}" = "1" ]; then
    PROFILER_FLAGS="--enable-diffusion-pipeline-profiler"
fi

mkdir -p "$LOG_DIR"
: > "$PID_FILE"

echo "Launching ${#PAIRS[@]} services (2rtx5090 deploy.sh config, 2 GPUs each)..."
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
    VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT="${VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT:-120}" \
    nohup vllm serve "$MODEL" \
      --omni --trust-remote-code \
      --host 0.0.0.0 --port "$port" \
      --num-gpus 2 \
      --num-weight-load-threads 8 \
      --tensor-parallel-size 2 \
      --text-encoder-tp-size 2 \
      --usp 1 --ring 1 \
      --quantization fp8 \
      --enable-cpu-offload \
      --diffusion-compile-granularity regional \
      --cache-backend cache_dit \
      --cache-config "$CACHE_CONFIG" \
      --enable-cache-dit-summary \
      --vae-patch-parallel-size 2 \
      --vae-parallel-mode tile --vae-use-tiling \
      --diffusion-attention-backend CUDNN_ATTN \
      $PROFILER_FLAGS \
      > "$log" 2>&1 &

    echo $! >> "$PID_FILE"
done

echo ""
echo "Launched ${#PAIRS[@]} services. PIDs: $(tr '\n' ' ' < "$PID_FILE")"
echo ""
echo "Tail a log:   tail -f $LOG_DIR/deploy_4svc.svc0.gpu0,1.log"
echo "Health check: for p in 0 1 2 3; do curl -s http://localhost:\$((9000+p))/health; echo; done"
echo "Host RAM:     free -g   (4-way needs ~480G of 503G — watch for OOM -9)"
echo "Concurrent benchmark:"
echo "  NUM_SERVICES=4 PORT_BASE=9000 bash ../../generate/generate.sh"
echo "Stop all:     kill \$(cat $PID_FILE)"
