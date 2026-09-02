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

# --- Request shape ---------------------------------------------------------
export TASK_TYPE="${TASK_TYPE:-fl2va}"
export WIDTH="${WIDTH:-832}"
export HEIGHT="${HEIGHT:-480}"
export DURATION="${DURATION:-5}"         # seconds

# --- Required inputs (defaults use inputs/i2va/ at the repo root) ----------
# The ONLY input knob is INPUT_DIR: each case is one directory holding
# prompt.txt plus 0-2 reference frame images (FL2VA's input, unlike Ref2VA's
# REFS list: 0 = text-only, 1 = first frame, 2 = first + last frame). All
# inputs are read from it — local files only, no URL download, and no
# PROMPT_FILE/FRAMES env overrides. Sorted filename order = upload order.
INPUT_DIR="${INPUT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)/inputs/i2va}"
export PROMPT_FILE="${INPUT_DIR}/prompt.txt"
export FRAMES="$(find "${INPUT_DIR}" -maxdepth 1 -type f ! -name 'prompt.txt' ! -name 'README*' | sort)"

# Echo the resolved inputs so a wrong case directory is immediately visible.
echo "[run.sh] prompt:  ${PROMPT_FILE}"
echo "[run.sh] frames:  ${FRAMES:-<none — text-only request>}"
echo

# --- Services ---------------------------------------------------------------
export PORT_BASE="${PORT_BASE:-9000}"
export NUM_SERVICES="${NUM_SERVICES:-1}"
export ROUNDS="${ROUNDS:-5}"

exec bash "$(dirname "$0")/generate.sh"
