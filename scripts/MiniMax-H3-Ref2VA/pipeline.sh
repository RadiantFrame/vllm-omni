#!/bin/bash
# One-terminal benchmark pipeline: deploy (background) -> wait healthy ->
# generate fan-out -> graceful stop.
#
# Usage:
#   bash scripts/MiniMax-H3/pipeline.sh                 # full run, stops server after
#   KEEP_SERVER=1 bash scripts/MiniMax-H3/pipeline.sh   # keep server up after generate
#   DEPLOY=... GENERATE=... PORT=... override the defaults below
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root, so paths work from anywhere

DEPLOY=${DEPLOY:-scripts/MiniMax-H3/deploy/rtx5090/4rtx5090/deploy.sh}
GENERATE=${GENERATE:-scripts/MiniMax-H3-Ref2VA/generate/generate.sh}
PORT=${PORT:-9000}
LOG=${LOG:-logs/pipeline_deploy.log}
HEALTH_TIMEOUT_MIN=${HEALTH_TIMEOUT_MIN:-15}   # model load + first init can take minutes

mkdir -p logs

echo "[pipeline] starting deploy: $DEPLOY (log: $LOG)"
bash "$DEPLOY" > "$LOG" 2>&1 &
DEPLOY_PID=$!
trap 'kill -TERM $DEPLOY_PID 2>/dev/null || true' EXIT   # never leak the server on Ctrl-C/error

echo "[pipeline] deploy pid=$DEPLOY_PID, waiting for http://localhost:$PORT/health ..."
deadline=$(( $(date +%s) + HEALTH_TIMEOUT_MIN * 60 ))
until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do
    if ! kill -0 "$DEPLOY_PID" 2>/dev/null; then
        echo "[pipeline] FATAL: deploy process died during startup — last log lines:" >&2
        tail -30 "$LOG" >&2
        exit 1
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "[pipeline] FATAL: health check timed out after ${HEALTH_TIMEOUT_MIN}min — tail $LOG" >&2
        exit 1
    fi
    sleep 5
done
echo "[pipeline] server healthy, launching generate: $GENERATE"

# PORT_BASE makes the fan-out client target the same port the deploy listens on.
PORT_BASE="$PORT" bash "$GENERATE"
gen_rc=$?

if [ "${KEEP_SERVER:-0}" = "1" ]; then
    echo "[pipeline] generate rc=$gen_rc; KEEP_SERVER=1 -> leaving server running (pid $DEPLOY_PID, log $LOG)"
else
    echo "[pipeline] generate rc=$gen_rc; stopping server gracefully (TERM -> $DEPLOY_PID)"
    kill -TERM "$DEPLOY_PID" 2>/dev/null || true
    # Wait for the launcher (and its workers) to wind down; hard-kill only as a
    # last resort — orphaned NCCL spin kernels burn GPUs until a driver reset.
    for _ in $(seq 1 24); do
        kill -0 "$DEPLOY_PID" 2>/dev/null || break
        sleep 5
    done
    if kill -0 "$DEPLOY_PID" 2>/dev/null; then
        echo "[pipeline] WARNING: server still alive after 2min, sending KILL" >&2
        kill -KILL "$DEPLOY_PID" 2>/dev/null || true
    fi
    echo "[pipeline] check clean exit: $(pgrep -af DiffusionWorker || echo 'no workers left')"
fi
exit "$gen_rc"
