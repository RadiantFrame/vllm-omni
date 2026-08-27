#!/usr/bin/env bash
set -euo pipefail

# Fan-out the 768p Ref2VA request (explicit width/height, mixed references:
# 3 images + 2 videos + 2 audios via repeated input_references fields) to N
# services CONCURRENTLY (one background curl per service), repeated for R
# rounds. Hardware-agnostic: services are identified purely by PORT/HOST —
# any vLLM-Omni video service works, regardless of which GPUs or deploy
# script started it. Results go to ./outputs with per-service/per-round names.
# Request shape: explicit WIDTH/HEIGHT (defaults 1344x768), task via TASK_TYPE
# (default ref2va; DURATION default 15 to match the current 768p benchmark;
# override with DURATION=8 for the 8s shape).
#
# H3 services need ~7 requests before inference latency converges in the
# server logs (req1 = compile warmup, req2 = lazy-init settling, steady from
# req3; a few configs need more). ROUNDS defaults to 7 so one invocation
# produces a converged per-round measurement series.
#
# Services are derived from PORT_BASE + NUM_SERVICES (contiguous ports):
#   bash generate.sh                              # 1 svc, 7 rounds
#   NUM_SERVICES=4 bash generate.sh               # 4 svc, 7 rounds
#   NUM_SERVICES=8 PORT_BASE=9000 bash generate.sh
# For non-contiguous ports, set PORTS explicitly (overrides base/count):
#   PORTS="9000 9002" bash generate.sh

# Services to hit: contiguous ports derived from PORT_BASE + NUM_SERVICES
# (defaults match the current deploy scripts: PORT_BASE=9000, 1 service).
# PORTS overrides both for explicit port lists.
PORT_BASE="${PORT_BASE:-9000}"
NUM_SERVICES="${NUM_SERVICES:-1}"
if [ -n "${PORTS:-}" ]; then
    read -r -a PORTS <<<"$PORTS"
else
    PORTS=()
    for ((i = 0; i < NUM_SERVICES; i++)); do
        PORTS+=("$((PORT_BASE + i))")
    done
fi

HOST="${HOST:-localhost}"
OUT_DIR="${OUT_DIR:-./outputs}"
ROUNDS="${ROUNDS:-7}"
SEED="${SEED:-0}"
DURATION="${DURATION:-15}"
TASK_TYPE="${TASK_TYPE:-ref2va}"
WIDTH="${WIDTH:-1344}"
HEIGHT="${HEIGHT:-768}"

mkdir -p "$OUT_DIR"

# Prompt is read from a local txt file (newlines preserved); required:
#   PROMPT_FILE=path/to/prompt.txt bash generate.sh
if [ -z "${PROMPT_FILE:-}" ]; then
    echo "ERROR: PROMPT_FILE must point to a prompt txt file" >&2
    exit 1
fi
[ -f "$PROMPT_FILE" ] || { echo "ERROR: PROMPT_FILE not found: $PROMPT_FILE" >&2; exit 1; }
PROMPT="$(cat "$PROMPT_FILE")"

# Reference inputs: uploaded as repeated "input_references" file fields
# (MiniMax-H3 mixed-reference path). The server auto-detects each file's
# modality from its MIME type/extension (.png/.jpg=image, .mp4/.mov=video,
# .wav/.mp3=audio) and enforces the Ref2VA contract (≤9 images, ≤3 videos,
# ≤3 audios, ≤12 total). REFS is required; image entries may be URLs
# (downloaded once), video/audio entries must be local files. E.g.:
#   REFS="a.png b.png c.png v1.mp4 v2.mp4 a1.wav a2.wav" \
#   bash generate.sh
if [ -z "${REFS:-}" ]; then
    echo "ERROR: REFS must list the reference files (space-separated; images may be URLs)" >&2
    exit 1
fi
declare -a REF_FILES=()
declare -a TEMP_FILES=()
for src in $REFS; do
    if [[ "$src" == http://* || "$src" == https://* ]]; then
        f="$(mktemp --suffix=.png)"
        TEMP_FILES+=("$f")
        echo "Downloading reference image: $src"
        curl -fsSL "$src" -o "$f"
        REF_FILES+=("$f")
    else
        [ -f "$src" ] || { echo "ERROR: reference file not found: $src" >&2; exit 1; }
        REF_FILES+=("$src")
    fi
done

cleanup() { [ ${#TEMP_FILES[@]} -gt 0 ] && rm -f "${TEMP_FILES[@]}"; }
trap cleanup EXIT

echo "Posting 768p/${DURATION}s ${TASK_TYPE} request to ${#PORTS[@]} service(s): ${PORTS[*]}, ${ROUNDS} round(s) (concurrent fan-out per round)..."
echo ""

fail=0
for ((r = 1; r <= ROUNDS; r++)); do
    echo "=== Round ${r}/${ROUNDS} ==="

    # Launch one background curl per service. Each writes its MP4 to a uniquely
    # named file and its HTTP status to a sidecar file.
    declare -a PID_ARR STAT_ARR OUT_ARR
    for i in "${!PORTS[@]}"; do
        port="${PORTS[$i]}"
        out="$OUT_DIR/${TASK_TYPE}_r${r}_svc${i}_port${port}_seed${SEED}.mp4"
        stat="$OUT_DIR/.status_r${r}_svc${i}_port${port}"
        : > "$stat"
        OUT_ARR[$i]="$out"
        STAT_ARR[$i]="$stat"

        # Build one -F flag per reference file (modality auto-detected server-side).
        REF_FLAGS=()
        for f in "${REF_FILES[@]}"; do REF_FLAGS+=(-F "input_references=@${f}"); done

        ( curl -sS -X POST "http://${HOST}:${port}/v1/videos/sync" \
            -F "prompt=${PROMPT}" \
            -F "fps=24" \
            -F "num_inference_steps=50" \
            -F "flow_shift=12" \
            -F "seed=${SEED}" \
            -F "width=${WIDTH}" \
            -F "height=${HEIGHT}" \
            -F 'extra_params={"task":"'"${TASK_TYPE}"'","duration":'"${DURATION}"',"audio_flow_shift":3.0}' \
            "${REF_FLAGS[@]}" \
            -o "$out" \
            -w '%{http_code}' > "$stat" ) &
        PID_ARR[$i]=$!
        echo "  service $i -> http://${HOST}:${port}  (pid ${PID_ARR[$i]})  -> $out"
    done

    echo ""
    echo "Waiting for all ${#PORTS[@]} request(s) of round ${r} to finish..."
    for i in "${!PORTS[@]}"; do
        if ! wait "${PID_ARR[$i]}"; then
            echo "  service $i (port ${PORTS[$i]}): curl exited non-zero" >&2
            fail=1
        fi
    done

    echo "Round ${r} results:"
    for i in "${!PORTS[@]}"; do
        port="${PORTS[$i]}"
        out="${OUT_ARR[$i]}"
        code="$(cat "${STAT_ARR[$i]}" 2>/dev/null || echo "?")"
        if [ "$code" = "200" ] && [ -s "$out" ]; then
            line="  [OK]  r${r} svc$i port=$port  -> $out"
            if command -v ffprobe >/dev/null 2>&1; then
                vc=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames,r_frame_rate -of csv=p=0 "$out" 2>/dev/null | head -1)
                line="$line  [$vc]"
            fi
            echo "$line"
        else
            echo "  [FAIL] r${r} svc$i port=$port  http=$code  (see $out for error body)"
            fail=1
        fi
        rm -f "${STAT_ARR[$i]}"
    done
    echo ""
done

if [ "$fail" = "1" ]; then exit 1; fi
echo "All done: ${ROUNDS} rounds x ${#PORTS[@]} service(s). Outputs in $OUT_DIR/ (${TASK_TYPE}_r<R>_svc<N>_...)"
echo "Read steady-state e2e_total_ms from each service log from round ~3 onward (rounds 1-2 are warmup/settling)."
