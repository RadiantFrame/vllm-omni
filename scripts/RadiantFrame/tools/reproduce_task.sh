#!/usr/bin/env bash

export PAPRIKA_KEY="${PAPRIKA_KEY:-}"
export TASK_ID="${TASK_ID:-}"

set -Eeuo pipefail
umask 077

usage() {
  cat <<'EOF'
Usage:
  PAPRIKA_KEY='...' scripts/reproduce_h3_task.sh TASK_ID [OUTPUT_DIR]
  PAPRIKA_KEY='...' TASK_ID='task_...' scripts/reproduce_h3_task.sh

Optional environment variables:
  PAPRIKA_BASE_URL API base URL (default: http://8.130.171.112)

The key may be a project API Key for a task in that project, or a system Master
Key for a read-only cross-project diagnostic export. The script never prints or
writes PAPRIKA_KEY to the export directory.
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 2 ]]; then
  usage >&2
  exit 64
fi

if [[ -z ${PAPRIKA_KEY:-} ]]; then
  echo "error: PAPRIKA_KEY is required" >&2
  exit 64
fi

for command_name in curl jq; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "error: required command not found: $command_name" >&2
    exit 69
  fi
done

task_id=${1:-${TASK_ID:-${REQUEST_ID:-}}}
if [[ -z $task_id ]]; then
  echo "error: TASK_ID, REQUEST_ID, or a positional task id is required" >&2
  exit 64
fi
if [[ ! $task_id =~ ^task_[A-Za-z0-9]+$ ]]; then
  echo "error: invalid task id: $task_id" >&2
  exit 64
fi

base_url=${PAPRIKA_BASE_URL:-http://8.130.171.112}
base_url=${base_url%/}
if [[ ! $base_url =~ ^https?://[^/]+(:[0-9]+)?$ ]]; then
  echo "error: PAPRIKA_BASE_URL must be an http(s) origin without a path" >&2
  exit 64
fi
if [[ $base_url == http://* ]]; then
  echo "warning: using the internal plaintext HTTP test endpoint" >&2
fi

output_dir=${2:-paprika-repro-$task_id}
mkdir -p "$output_dir/inputs" "$output_dir/outputs"
chmod 700 "$output_dir" "$output_dir/inputs" "$output_dir/outputs"

manifest_tmp=$(mktemp "${TMPDIR:-/tmp}/paprika-repro.XXXXXX")
trap 'rm -f "$manifest_tmp"' EXIT

manifest_url="$base_url/v1/minimax-h3/requests/$task_id/reproduction"
if ! curl --fail-with-body --silent --show-error \
  --header "Authorization: Key $PAPRIKA_KEY" \
  --header "Accept: application/json" \
  --output "$manifest_tmp" \
  "$manifest_url"; then
  if [[ -s $manifest_tmp ]]; then
    jq . "$manifest_tmp" >&2 2>/dev/null || sed -n '1,120p' "$manifest_tmp" >&2
  fi
  exit 1
fi

if ! jq -e --arg task_id "$task_id" '.request_id == $task_id' "$manifest_tmp" >/dev/null; then
  echo "error: API returned an invalid reproduction manifest" >&2
  jq . "$manifest_tmp" >&2 2>/dev/null || true
  exit 1
fi

install -m 600 "$manifest_tmp" "$output_dir/task.json"
jq -r '.prompt' "$manifest_tmp" >"$output_dir/prompt.txt"
jq '.parameters' "$manifest_tmp" >"$output_dir/parameters.json"
if jq -e '.context_ir_optimized_prompt != null' "$manifest_tmp" >/dev/null; then
  jq -r '.context_ir_optimized_prompt' "$manifest_tmp" >"$output_dir/context-ir-prompt.txt"
fi

asset_extension() {
  case "$1" in
    image/png) echo ".png" ;;
    image/jpeg) echo ".jpg" ;;
    image/webp) echo ".webp" ;;
    video/mp4) echo ".mp4" ;;
    audio/mpeg) echo ".mp3" ;;
    audio/wav|audio/x-wav) echo ".wav" ;;
    audio/mp4) echo ".m4a" ;;
    *) echo "" ;;
  esac
}

asset_count=0
while IFS=$'\t' read -r asset_id role kind mime_type download_url source_url; do
  [[ -n $asset_id ]] || continue
  asset_count=$((asset_count + 1))
  safe_role=$(printf '%s' "${role:-asset}" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9._-' '_')
  safe_id=$(printf '%s' "$asset_id" | tr -c 'A-Za-z0-9._-' '_')
  extension=$(asset_extension "$mime_type")
  destination="$output_dir/inputs/$(printf '%02d' "$asset_count")_${safe_role}_${safe_id}${extension}"
  if [[ -n $download_url ]]; then
    if [[ ! $asset_id =~ ^[A-Za-z0-9_.-]+$ ]]; then
      echo "error: invalid asset id in manifest: $asset_id" >&2
      exit 1
    fi
    curl --fail-with-body --silent --show-error \
      --header "Authorization: Key $PAPRIKA_KEY" \
      --output "$destination" \
      "$base_url/v1/minimax-h3/requests/$task_id/input-assets/$asset_id"
  elif [[ -n $source_url ]]; then
    curl --fail-with-body --silent --show-error --location \
      --output "$destination" \
      "$source_url"
  else
    echo "warning: asset $asset_id has no downloadable copy" >&2
  fi
done < <(
  jq -r '
    [.input_assets[], .reference_assets[]]
    | unique_by(.asset_id)
    | .[]
    | [.asset_id, .role, .kind, (.mime_type // ""), (.download_url // ""), (.source_url // "")]
    | @tsv
  ' "$manifest_tmp"
)

video_download_url=$(jq -r '.video_download_url // empty' "$manifest_tmp")
video_source_url=$(jq -r '.video_source_url // empty' "$manifest_tmp")
if [[ -n $video_download_url ]]; then
  curl --fail-with-body --silent --show-error \
    --header "Authorization: Key $PAPRIKA_KEY" \
    --output "$output_dir/outputs/video.mp4" \
    "$base_url/v1/minimax-h3/requests/$task_id/video"
elif [[ -n $video_source_url ]]; then
  curl --fail-with-body --silent --show-error --location \
    --output "$output_dir/outputs/video.mp4" \
    "$video_source_url"
fi

status=$(jq -r '.status' "$manifest_tmp")
capability=$(jq -r '.capability' "$manifest_tmp")
echo "exported $task_id ($capability, $status) to $output_dir"
echo "prompt: $output_dir/prompt.txt"
echo "manifest: $output_dir/task.json"
echo "input assets: $asset_count"
if [[ -f $output_dir/outputs/video.mp4 ]]; then
  echo "video: $output_dir/outputs/video.mp4"
else
  echo "video: not available for task status $status"
fi
