#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper around generate.sh: set request parameters via environment
# variables here (or rely on these defaults), then fan out. Everything not
# set below is inherited by generate.sh's own defaults (see that script for
# the full list: HOST, SEED, PORTS, ROUNDS, ...).
#
# Usage:
#   bash run.sh                                    # 480p/5s defaults below
#   WIDTH=1344 HEIGHT=768 DURATION=15 bash run.sh  # the 768p benchmark shape
#   INPUT_DIR=inputs/i2va bash run.sh              # another case directory
#   FRAMES="" bash run.sh                          # text-only (0 frames)
#   FRAMES="first.png last.png" bash run.sh        # first + last frame

# --- Request shape ---------------------------------------------------------
export TASK_TYPE="${TASK_TYPE:-fl2va}"
export WIDTH="${WIDTH:-832}"
export HEIGHT="${HEIGHT:-480}"
export DURATION="${DURATION:-5}"         # seconds

# --- Required inputs (defaults use inputs/i2va/ at the repo root) ----------
# Each case is one directory holding prompt.txt plus 0-2 reference frame
# images (FL2VA's input, unlike Ref2VA's REFS list: 0 = text-only, 1 = first
# frame, 2 = first + last frame). All must be local files — no URL download.
# PROMPT_FILE: local txt file, newlines preserved.
# FRAMES: the reference frame images, in upload order (first, then last);
# defaults to INPUT_DIR images (files other than prompt.txt/README) in
# sorted filename order — an empty result means text-only.
INPUT_DIR="${INPUT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)/inputs/i2va}"
export PROMPT_FILE="${PROMPT_FILE:-${INPUT_DIR}/prompt.txt}"
export FRAMES="${FRAMES:-$(find "${INPUT_DIR}" -maxdepth 1 -type f ! -name 'prompt.txt' ! -name 'README*' | sort)}"

# Echo the resolved inputs so a wrong case directory is immediately visible.
echo "[run.sh] prompt:  ${PROMPT_FILE}"
echo "[run.sh] frames:  ${FRAMES:-<none — text-only request>}"
echo

# --- Services ---------------------------------------------------------------
export PORT_BASE="${PORT_BASE:-9000}"
export NUM_SERVICES="${NUM_SERVICES:-1}"
export ROUNDS="${ROUNDS:-5}"

exec bash "$(dirname "$0")/generate.sh"
