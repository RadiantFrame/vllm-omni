#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper around generate.sh: set request parameters via environment
# variables here (or rely on these defaults), then fan out. Everything not
# set below is inherited by generate.sh's own defaults (see that script for
# the full list: HOST, SEED, PORTS, ROUNDS, OUT_DIR, ...).
#
# Usage:
#   bash run.sh                                    # use the defaults below
#   WIDTH=832 HEIGHT=480 DURATION=5 bash run.sh    # override on the CLI
#
# Required (no defaults): PROMPT_FILE, REFS.

# --- Request shape ---------------------------------------------------------
export TASK_TYPE="${TASK_TYPE:-ref2va}"   # ref2va | fl2va | t2va
export WIDTH="${WIDTH:-1344}"
export HEIGHT="${HEIGHT:-768}"
export DURATION="${DURATION:-15}"         # seconds

# --- Required inputs (defaults use inputs/r2va/ at the repo root) ----------
# PROMPT_FILE: local txt file, newlines preserved.
# REFS: space-separated reference files (images may be URLs), e.g.
#   REFS="a.png b.png c.png v1.mp4 v2.mp4 a1.wav a2.wav"
INPUT_DIR="${INPUT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)/inputs/r2va}"
export PROMPT_FILE="${PROMPT_FILE:-${INPUT_DIR}/prompt.txt}"
# IMPORTANT: the server maps references to prompt tags strictly by UPLOAD
# ORDER (<Picture 1..N> / <Video 1..N> follow the order of the fields below).
# The default collects INPUT_DIR files in sorted filename order — if that does
# not match the order in the prompt, pass REFS explicitly in the prompt order:
#   REFS="pic1.jpeg pic2.jpeg pic3.jpeg video1.mp4 ..."
export REFS="${REFS:-"$(ls "${INPUT_DIR}" | grep -Ev '^(prompt\.txt|README)' | sort | sed "s|^|${INPUT_DIR}/|")"}"

# Echo the index -> file mapping so mismatches with the prompt are visible.
echo "[run.sh] reference order (defines <Picture N>/<Video N> numbering):"
idx=1
for f in $REFS; do echo "  [$idx] $f"; idx=$((idx + 1)); done
echo

# --- Services ---------------------------------------------------------------
export PORT_BASE="${PORT_BASE:-9000}"
export NUM_SERVICES="${NUM_SERVICES:-1}"
export ROUNDS="${ROUNDS:-5}"

exec bash "$(dirname "$0")/generate.sh"
