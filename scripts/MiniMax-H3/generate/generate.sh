#!/usr/bin/env bash
set -euo pipefail

# Fan-out the FL2VA request (single first-frame keyframe + prompt) to N
# services CONCURRENTLY (one background curl per service), repeated for R
# rounds. Hardware-agnostic: services are identified purely by PORT/HOST —
# any vLLM-Omni video service works, regardless of which GPUs or deploy
# script started it. Results go to ./outputs with per-service/per-round names.
#
# Input difference vs Ref2VA's generate.sh: FL2VA takes 0, 1, or 2 local
# reference FRAME images (0 = pure text-to-video, 1 = first frame, 2 = first
# + last frame), uploaded as repeated "input_reference" file fields — not a
# mixed REFS list, and never URL downloads.
#
# Request shape: explicit WIDTH/HEIGHT (defaults 832x480), like the Ref2VA
# generate.sh.
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
DURATION="${DURATION:-5}"
TASK_TYPE="${TASK_TYPE:-fl2va}"
WIDTH="${WIDTH:-832}"
HEIGHT="${HEIGHT:-480}"
# Optional explicit frame-index mapping (see the FRAMES comment above).
FRAME_INDICES="${FRAME_INDICES:-}"

mkdir -p "$OUT_DIR"

# Prompt is read from a local txt file (newlines preserved); required:
#   PROMPT_FILE=path/to/prompt.txt bash generate.sh
if [ -z "${PROMPT_FILE:-}" ]; then
    echo "ERROR: PROMPT_FILE must point to a prompt txt file" >&2
    exit 1
fi
[ -f "$PROMPT_FILE" ] || { echo "ERROR: PROMPT_FILE not found: $PROMPT_FILE" >&2; exit 1; }
PROMPT="$(cat "$PROMPT_FILE")"

# Reference frames: LOCAL image files uploaded as repeated "input_reference"
# file fields. The server distinguishes roles by COUNT + upload order
# (pipeline _resolve_fl2va_keyframe_indices): 1 image defaults to frame
# indices [0] (first frame), 2 images to [0, -1] (first image = first frame,
# second image = last frame). 0 frames = text-only. Max 2. E.g.:
#   FRAMES="" bash generate.sh                        # 0 frames (text-only)
#   FRAMES="first.png" bash generate.sh               # first frame only
#   FRAMES="first.png last.png" bash generate.sh      # first + last frame
# FRAME_INDICES optionally overrides the mapping via extra_params (one of
# [0], [-1], [0,-1]; must match the image count): e.g. a single image used
# as the LAST frame:
#   FRAMES="last.png" FRAME_INDICES="-1" bash generate.sh
FRAME_FILES=()
for src in ${FRAMES:-}; do
    [ -f "$src" ] || { echo "ERROR: reference frame not found: $src" >&2; exit 1; }
    FRAME_FILES+=("$src")
done
if [ "${#FRAME_FILES[@]}" -gt 2 ]; then
    echo "ERROR: FRAMES accepts at most 2 files (first [+ last]), got ${#FRAME_FILES[@]}" >&2
    exit 1
fi
case "${#FRAME_FILES[@]}" in
    0) frames_desc="0 reference frames (text-only)" ;;
    1) frames_desc="first frame only" ;;
    2) frames_desc="first + last frame" ;;
esac

# Optional explicit frame_indices (server default: [0] for 1 image, [0,-1]
# for 2). Validated server-side against the image count.
EXTRA_FRAME_INDICES=""
if [ -n "$FRAME_INDICES" ]; then
    if [ "${#FRAME_FILES[@]}" -eq 0 ]; then
        echo "ERROR: FRAME_INDICES requires at least one frame in FRAMES" >&2
        exit 1
    fi
    EXTRA_FRAME_INDICES=',"frame_indices":['"$FRAME_INDICES"']'
fi

echo "Posting ${WIDTH}x${HEIGHT}/${DURATION}s ${TASK_TYPE} request (${frames_desc}) to ${#PORTS[@]} service(s): ${PORTS[*]}, ${ROUNDS} round(s) (concurrent fan-out per round)..."
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

        # Build one -F flag per reference frame (order: first, then last).
        FRAME_FLAGS=()
        for f in "${FRAME_FILES[@]}"; do FRAME_FLAGS+=(-F "input_reference=@${f}"); done

        ( curl -sS -X POST "http://${HOST}:${port}/v1/videos/sync" \
            -F "prompt=${PROMPT}" \
            -F "fps=24" \
            -F "num_inference_steps=50" \
            -F "flow_shift=12" \
            -F "seed=${SEED}" \
            -F "width=${WIDTH}" \
            -F "height=${HEIGHT}" \
            -F 'extra_params={"task":"'"${TASK_TYPE}"'","duration":'"${DURATION}"',"audio_flow_shift":3.0'"${EXTRA_FRAME_INDICES}"'}' \
            "${FRAME_FLAGS[@]}" \
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
