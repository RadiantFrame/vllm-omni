#!/usr/bin/env bash
set -euo pipefail

# Call the running vLLM-Omni MiniMax-H3 FL2VA service (deploy.sh, port 8000)
# with the same H3-Context-IR prompt + first-frame keyframe + seed as
# ../sglang/scripts/MiniMax-H3/h800/generate.sh.
#
# sglang vs vLLM-Omni differences:
#   - sglang: JSON body -> POST /v1/videos (async) + poll + GET .../content, port 30010
#   - vLLM-Omni: multipart form -> POST /v1/videos/sync (returns raw MP4), port 8000
#   - keyframe: sglang sends a CDN URL; vLLM-Omni's input_reference is a file upload,
#     so we download the CDN image to a temp file first (or set FIRST_FRAME to a local
#     path to skip the download).
#   - fl2va task / duration / audio_flow_shift go in extra_params (per the H3 recipe).

BASE_URL="${BASE_URL:-http://localhost:8000}"
API_URL="${API_URL:-${BASE_URL}/v1/videos/sync}"
OUTPUT="${OUTPUT:-fl2va.mp4}"
SEED="${SEED:-0}"
DURATION="${DURATION:-5}"
KEYFRAME_URL="${KEYFRAME_URL:-https://cdn.hailuoai.com/prod/hailuo_demo/testsets/H3_AA_I2VA/gallery/sr_v17_variants_seed42_43_20260724/inputs/4a3a90bf9100_KDmcbkhzYo5sjjxr9FqcVmWVnzb.png}"

# Same H3-Context-IR prompt as the sglang script (newlines preserved).
PROMPT=$(cat <<'EOF'
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] This is a live-action, cinematic shot with a shallow depth of field. The camera holds a perfectly static shot throughout the entire eight-second duration, capturing a cozy family gathering in a traditional Japanese dining room. The scene opens with a large, intricately patterned blue and white ceramic bowl of ramen in the immediate foreground, rendered in crisp, sharp focus. The bowl sits on a smooth, polished long wooden table. Inside the bowl, a rich, oily golden-brown broth surrounds yellow wavy noodles, topped with two thick, round slices of chashu pork featuring visible fat marbling and a distinct spiral meat pattern. A generous mound of freshly chopped, bright green scallions rests in the center, and a crisp, dark green rectangular sheet of nori seaweed is tucked into the right edge. To the left of the bowl, a pair of light brown wooden chopsticks rests horizontally on a small, dark rectangular chopstick rest, near a small cylindrical ceramic teacup with blue painted patterns. On the right side of the table, a spherical paper lantern with a ribbed bamboo frame sits on a black wooden base. In the background, a large family of seven is gathered around the table, initially appearing as a soft, blurred presence. Behind them, traditional Japanese sliding shoji screens with wooden lattice frames are open, revealing a bright outdoor scene with lush green trees. Early in the clip, the thick, white steam rising from the hot ramen broth immediately intensifies, billowing upwards in thick, swirling clouds that dance continuously above the bowl. As the clip progresses into the middle seconds, the camera maintains its static position while the focus begins a deliberate, smooth shift deeper into the room. The foreground ramen bowl, its vibrant ingredients, and the rising steam gradually soften into a hazy, out-of-focus blur. Simultaneously, the family members in the background come into sharp, detailed clarity. The heavy steam continues to rise from the foreground, creating a dynamic, translucent veil between the camera and the family. With the focus now firmly locked on the background, the vibrant family dinner comes alive. The man in the dark navy blue long-sleeved shirt on the left leans forward, his mouth moving animatedly in a silent exchange. The young girl in the crisp white short-sleeved t-shirt beside him smiles brightly, looking toward the center of the table. The woman on the far left, wearing a soft light blue long-sleeved blouse, turns her head slightly, smiling gently. Across the table, the woman in the light grey button-down shirt smiles broadly, her eyes crinkling, as she rests her hands near her plate. The woman in the dark grey top further back uses her wooden chopsticks to pick up a small piece of food from a central ceramic dish filled with bright red pickled vegetables. The woman in the center back in the light grey sweater smiles gently, her hands clasped softly in front of her, observing the interaction. Throughout the remainder of the clip, the family continues their lively physical interaction, their mouths moving in continuous, silent cadences of conversation, while the thick, white steam from the blurred ramen bowl in the foreground never stops rising, adding a comforting atmosphere to the warm gathering.

overall_soundscape: The soundscape begins with a quiet room tone mixed with the faint, airy rustle of the thick steam billowing from the hot ramen bowl in the foreground, accompanied by the subtle, continuous hissing and bubbling of the rich broth. As the visual focus shifts deeper into the room, the physical sounds of the bustling family dinner become dominant in the foreground. The clear, sharp clinking of ceramic bowls and wooden chopsticks touching plates is clearly heard as the family members reach for food. This is followed by the faint, muffled thud of a cup being set down on the smooth wooden table, and the subtle, rhythmic rustle of cotton and wool clothing as the family members lean forward and gesture, perfectly capturing the lively, physical atmosphere of the shared meal.

non_diegetic_music: A gentle, heartwarming acoustic guitar melody plays softly in the background, accompanied by the subtle, resonant notes of a traditional Japanese koto. The music maintains a slow, comforting tempo that enhances the cozy, nostalgic, and joyful atmosphere of the family gathering.
EOF
)

# Resolve the first-frame keyframe (vLLM-Omni needs a file upload).
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

echo "Posting FL2VA request to $API_URL (seed=$SEED, duration=${DURATION}s)..."
http_code=$(curl -sS -X POST "$API_URL" \
    -F "prompt=${PROMPT}" \
    -F "fps=24" \
    -F "num_inference_steps=50" \
    -F "flow_shift=12" \
    -F "seed=${SEED}" \
    -F "width=832" \
    -F "height=480" \
    -F "extra_params={\"task\":\"fl2va\",\"duration\":${DURATION},\"audio_flow_shift\":3.0}" \
    -F "input_reference=@${keyframe};type=image/png" \
    -o "$OUTPUT" \
    -w '%{http_code}')

if [ "$http_code" != "200" ]; then
    echo "Request failed (HTTP $http_code). Response body:" >&2
    cat "$OUTPUT" >&2 || true
    exit 1
fi

echo "Saved video -> $OUTPUT"
if command -v ffprobe >/dev/null 2>&1; then
    vcodec=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 "$OUTPUT" 2>/dev/null || true)
    acodec=$(ffprobe -v error -select_streams a:0 -show_entries stream=codec_name -of csv=p=0 "$OUTPUT" 2>/dev/null || true)
    echo "streams: video=${vcodec:-none} audio=${acodec:-none}"
fi
