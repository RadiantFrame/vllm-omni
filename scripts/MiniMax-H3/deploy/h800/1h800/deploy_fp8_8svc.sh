#!/bin/bash
# Launch 8 independent MiniMax-H3 FL2VA services on an 8xH800 machine.
# Each service = deploy_fp8.sh config (1-GPU cpu-offload + FP8 + Cache-DiT)
# per GPU -- the FP8 throughput counterpart of deploy_8svc.sh (BF16).
#
#   service 0 -> GPU 0 -> port 9000 (master 29500)
#   ...
#   service 7 -> GPU 7 -> port 9007 (master 29507)
#
# Per-service performance == deploy_fp8.sh measured numbers (480p/832x480/5s:
# 50.5s steady, 36.3 GiB resident). 8x aggregate for throughput.
# FP8 advantages over the BF16 8svc (deploy_8svc.sh, 61.5s/req measured):
#   - swaps halve (31G vs 62G per direction over shared PCIe)
#   - FP8 GEMM during denoise
#   - host RAM per service ~65G (vs ~130G BF16) -> 8x65 ~= 520G, comfortable
# Expected: lower contention than BF16's +16.5% and higher aggregate throughput
# (the BF16 8svc layout measured 7.61 videos/min with the last-round-max
# metric; re-measure this one the same way before comparing).
#
# All 8 start concurrently in the background; per-service logs go to
# deploy_fp8_8svc.svc<N>.gpu<N>.log in LOG_DIR. PIDs are written to
# deploy_fp8_8svc.pids for `kill $(cat ...)`.
#
# MASTER_ADDR/PORT are pinned per service (29500+i): without them the
# torch.distributed TCP store picks a RANDOM port and concurrent launches
# can race into EADDRINUSE (observed once on the BF16 8-way launcher).
# FP8+cpu-offload requires upstream #5910 (stream-offload online quant):
# on older code the load-time quant accumulation OOMs each 80G card.

set -euo pipefail

MODEL=${MODEL:-/data/models/modelscope/MiniMax/MiniMax-H3/FL2VA}
LOG_DIR="${LOG_DIR:-./logs}"
PORT_BASE="${PORT_BASE:-9000}"
PID_FILE="$LOG_DIR/deploy_fp8_8svc.pids"

# One GPU per service, 8 services.
GPUS=(0 1 2 3 4 5 6 7)

# Cache-DiT H3 "high" profile (same as deploy_fp8.sh).
CACHE_CONFIG='{"Fn_compute_blocks":1,"Bn_compute_blocks":0,"max_warmup_steps":4,"residual_diff_threshold":0.04,"max_continuous_cached_steps":1,"enable_taylorseer":false}'

PROFILER_FLAGS=""
if [ "${PROFILER:-0}" = "1" ]; then
    PROFILER_FLAGS="--enable-diffusion-pipeline-profiler"
fi

mkdir -p "$LOG_DIR"
: > "$PID_FILE"

echo "Launching ${#GPUS[@]} services (1h800 deploy_fp8.sh config, 1 GPU each)..."
for i in "${!GPUS[@]}"; do
    gpu="${GPUS[$i]}"
    port=$((PORT_BASE + i))
    master_port=$((29500 + i))
    log="$LOG_DIR/deploy_fp8_8svc.svc${i}.gpu${gpu}.log"
    echo "  service $i: GPU=$gpu  port=$port  master_port=$master_port  log=$log"

    CUDA_VISIBLE_DEVICES="$gpu" \
    MASTER_ADDR=127.0.0.1 \
    MASTER_PORT="$master_port" \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
    VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT=300 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup vllm serve "$MODEL" \
      --omni --trust-remote-code \
      --host 0.0.0.0 --port "$port" \
      --num-gpus 1 \
      --enable-cpu-offload \
      --quantization fp8 \
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
echo "Tail a log:   tail -f $LOG_DIR/deploy_fp8_8svc.svc0.gpu0.log"
echo "Health check: for p in \$(seq 0 7); do curl -s http://localhost:\$((9000+p))/health; echo; done"
echo "Benchmark (7 rounds x 8 lanes, converged steady state from round ~3):"
echo "  NUM_SERVICES=8 PORT_BASE=9000 ROUNDS=7 bash ../../generate/generate_480p_5s_nsvc.sh"
echo "Stop all:     kill \$(cat $PID_FILE)"
