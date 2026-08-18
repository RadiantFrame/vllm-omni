#!/usr/bin/env bash
set -euo pipefail

# Fan-out the 480p/5s FL2VA request to N services CONCURRENTLY (one background
# curl per service). Hardware-agnostic: services are identified purely by
# PORT/HOST — any vLLM-Omni video service works, regardless of which GPUs or
# deploy script started it. Results go to ./outputs with per-service names.
#
# Based on generate_480p_5s.sh (same prompt / keyframe / params).
# Services are derived from PORT_BASE + NUM_SERVICES (contiguous ports):
#   bash generate_480p_5s_n.sh                          # 2 svc: 8000-8001 (default)
#   NUM_SERVICES=4 bash generate_480p_5s_n.sh           # 4 svc: 8000-8003
#   PORT_BASE=9000 NUM_SERVICES=3 bash generate_480p_5s_n.sh  # 3 svc: 9000-9002
# For non-contiguous ports, set PORTS explicitly (overrides base/count):
#   PORTS="8000 8002" bash generate_480p_5s_n.sh

# Services to hit: contiguous ports derived from PORT_BASE + NUM_SERVICES
# (defaults match deploy/rtx5090/2rtx5090/deploy_tier0_2svc.sh:
# PORT_BASE=8000, 2 services). PORTS overrides both for explicit port lists.
PORT_BASE="${PORT_BASE:-8000}"
NUM_SERVICES="${NUM_SERVICES:-2}"
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
SEED="${SEED:-0}"
DURATION="${DURATION:-5}"
KEYFRAME_URL="${KEYFRAME_URL:-https://cdn.hailuoai.com/prod/hailuo_demo/testsets/H3_AA_I2VA/gallery/sr_v17_variants_seed42_43_20260724/inputs/4a3a90bf9100_KDmcbkhzYo5sjjxr9FqcVmWVnzb.png}"

mkdir -p "$OUT_DIR"

# Same H3-Context-IR prompt as generate_480p_5s.sh (newlines preserved).
PROMPT=$(cat <<'EOF'
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] This is a live-action, cinematic shot with a shallow depth of field. The camera holds a perfectly static shot throughout the entire eight-second duration, capturing a cozy family gathering in a traditional Japanese dining room. The scene opens with a large, intricately patterned blue and white ceramic bowl of ramen in the immediate foreground, rendered in crisp, sharp focus. The bowl sits on a smooth, polished long wooden table. Inside the bowl, a rich, oily golden-brown broth surrounds yellow wavy noodles, topped with two thick, round slices of chashu pork featuring visible fat marbling and a distinct spiral meat pattern. A generous mound of freshly chopped, bright green scallions rests in the center, and a crisp, dark green rectangular sheet of nori seaweed is tucked into the right edge. To the left of the bowl, a pair of light brown wooden chopsticks rests horizontally on a small, dark rectangular chopstick rest, near a small cylindrical ceramic teacup with blue painted patterns. On the right side of the table, a spherical paper lantern with a ribbed bamboo frame sits on a black wooden base. In the background, a large family of seven is gathered around the table, initially appearing as a soft, blurred presence. Behind them, traditional Japanese sliding shoji screens with wooden lattice frames are open, revealing a bright outdoor scene with lush green trees. Early in the clip, the thick, white steam rising from the hot ramen broth immediately intensifies, billowing upwards in thick, swirling clouds that dance continuously above the bowl. As the clip progresses into the middle seconds, the camera maintains its static position while the focus begins a deliberate, smooth shift deeper into the room. The foreground ramen bowl, its vibrant ingredients, and the rising steam gradually soften into a hazy, out-of-focus blur. Simultaneously, the family members in the background come into sharp, detailed clarity. The heavy steam continues to rise from the foreground, creating a dynamic, translucent veil between the camera and the family. With the focus now firmly locked on the background, the vibrant family dinner comes alive. The man in the dark navy blue long-sleeved shirt on the left leans forward, his mouth moving animatedly in a silent exchange. The young girl in the crisp white short-sleeved t-shirt beside him smiles brightly, looking toward the center of the table. The woman on the far left, wearing a soft light blue long-sleeved blouse, turns her head slightly, smiling gently. Across the table, the woman in the light grey button-down shirt smiles broadly, her eyes crinkling, as she rests her hands near her plate. The woman in the dark grey top further back uses her wooden chopsticks to pick up a small piece of food from a central ceramic dish filled with bright red pickled vegetables. The woman in the center back in the light grey sweater smiles gently, her hands clasped softly in front of her, observing the interaction. Throughout the remainder of the clip, the family continues their lively physical interaction, their mouths moving in continuous, silent cadences of conversation, while the thick, white steam from the blurred ramen bowl in the foreground never stops rising, adding a comforting atmosphere to the warm gathering.

overall_soundscape: The soundscape begins with a quiet room tone mixed with the faint, airy rustle of the thick steam billowing from the hot ramen bowl in the foreground, accompanied by the subtle, continuous hissing and bubbling of the rich broth. As the visual focus shifts deeper into the room, the physical sounds of the bustling family dinner become dominant in the foreground. The clear, sharp clinking of ceramic bowls and wooden chopsticks touching plates is clearly heard as the family members reach for food. This is followed by the faint, muffled thud of a cup being set down on the smooth wooden table, and the subtle, rhythmic rustle of cotton and wool clothing as the family members lean forward and gesture, perfectly capturing the lively, physical atmosphere of the shared meal.

non_diegetic_music: A gentle, heartwarming acoustic guitar melody plays softly in the background, accompanied by the subtle, resonant notes of a traditional Japanese koto. The music maintains a slow, comforting tempo that enhances the cozy, nostalgic, and joyful atmosphere of the family gathering.
EOF
)

# Resolve the first-frame keyframe ONCE (shared by all services).
created_temp=0
if [ -n "${FIRST_FRAME:-}" ] && [ -f "${FIRST_FRAME}" ]; then
    keyframe="$FIRST_FRAME"
else
    keyframe="$(mktemp --suffix=.png)"
    created_temp=1
    echo "Downloading keyframe: $KEYFRAME_URL"
    curl -fsSL "$KEYFRAME_URL" -o "$keyframe"
fi
cleanup() { [ "$created_temp" = "1" ] && rm -f "$keyframe"; }
trap cleanup EXIT

echo "Posting 480p/5s FL2VA request to ${#PORTS[@]} service(s): ${PORTS[*]} (concurrent)..."
echo ""

# Launch one background curl per service. Each writes its MP4 to a uniquely
# named file and its HTTP status to a sidecar file.
declare -a PID_ARR STAT_ARR OUT_ARR
for i in "${!PORTS[@]}"; do
    port="${PORTS[$i]}"
    out="$OUT_DIR/fl2va_svc${i}_port${port}_seed${SEED}.mp4"
    stat="$OUT_DIR/.status_svc${i}_port${port}"
    : > "$stat"
    OUT_ARR[$i]="$out"
    STAT_ARR[$i]="$stat"

    ( curl -sS -X POST "http://${HOST}:${port}/v1/videos/sync" \
        -F "prompt=${PROMPT}" \
        -F "fps=24" \
        -F "num_inference_steps=50" \
        -F "flow_shift=12" \
        -F "seed=${SEED}" \
        -F "width=832" \
        -F "height=480" \
        -F "extra_params={\"task\":\"fl2va\",\"duration\":${DURATION},\"audio_flow_shift\":3.0}" \
        -F "input_reference=@${keyframe};type=image/png" \
        -o "$out" \
        -w '%{http_code}' > "$stat" ) &
    PID_ARR[$i]=$!
    echo "  service $i -> http://${HOST}:${port}  (pid ${PID_ARR[$i]})  -> $out"
done

echo ""
echo "Waiting for all ${#PORTS[@]} request(s) to finish..."
fail=0
for i in "${!PORTS[@]}"; do
    if ! wait "${PID_ARR[$i]}"; then
        echo "  service $i (port ${PORTS[$i]}): curl exited non-zero" >&2
        fail=1
    fi
done

echo ""
echo "Results:"
for i in "${!PORTS[@]}"; do
    port="${PORTS[$i]}"
    out="${OUT_ARR[$i]}"
    code="$(cat "${STAT_ARR[$i]}" 2>/dev/null || echo "?")"
    if [ "$code" = "200" ] && [ -s "$out" ]; then
        line="  [OK]  svc$i port=$port  -> $out"
        if command -v ffprobe >/dev/null 2>&1; then
            vc=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames,r_frame_rate -of csv=p=0 "$out" 2>/dev/null | head -1)
            line="$line  [$vc]"
        fi
        echo "$line"
    else
        echo "  [FAIL] svc$i port=$port  http=$code  (see $out for error body)"
        fail=1
    fi
    rm -f "${STAT_ARR[$i]}"
done

if [ "$fail" = "1" ]; then exit 1; fi
echo ""
echo "All done. Outputs in $OUT_DIR/"
