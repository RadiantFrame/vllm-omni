#!/usr/bin/env python3
"""Fan-out the FL2VA request to N services concurrently, R rounds.

Python port of scripts/MiniMax-H3/generate/generate.sh (drop-in replacement:
same env knobs, same request shape, same output naming).

Env knobs (defaults match the bash version unless noted):
  PORT_BASE      base port for contiguous derivation        (9000)
  NUM_SERVICES   number of services                          (1)
  PORTS          explicit space-separated port list; overrides base/count
  HOST           service host                                 (localhost)
  OUT_DIR        output directory                             (./outputs)
  ROUNDS         rounds of concurrent fan-out                 (7)
  SEED           generation seed                              (0)
  TASK_TYPE      extra_params task                            (fl2va)
  DURATION       audio/video seconds in extra_params          (5)
  WIDTH          explicit output width                        (832)
  HEIGHT         explicit output height                       (480)
  INPUT_DIR      the ONLY input knob: per-case directory holding prompt.txt
                 plus 0-2 reference frame images (0 = text-only, 1 = first
                 frame, 2 = first + last frame, sorted filename order =
                 upload order). Default: <repo>/inputs/i2va. No URL
                 download; PROMPT_FILE/FRAMES env overrides do not exist.
  REQUEST_TIMEOUT  per-request read timeout, seconds         (1800)

Notes:
- The request is multipart form -> POST http://HOST:PORT/v1/videos/sync,
  exactly mirroring the curl -F fields of the bash client, including the
  repeated "input_reference" file fields (one per frame).
- Per-request wall time is measured client-side (in addition to the
  server-side e2e_total_ms you can grep from service logs).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

# ---------------------------------------------------------------------------
# configuration (env parity with the bash fan-out client)
# ---------------------------------------------------------------------------

PORT_BASE = int(os.environ.get("PORT_BASE", "9000"))
NUM_SERVICES = int(os.environ.get("NUM_SERVICES", "1"))
HOST = os.environ.get("HOST", "localhost")
OUT_DIR = os.environ.get("OUT_DIR", "./outputs")
ROUNDS = int(os.environ.get("ROUNDS", "5"))
SEED = os.environ.get("SEED", "0")
TASK_TYPE = os.environ.get("TASK_TYPE", "fl2va")
DURATION = os.environ.get("DURATION", "5")
WIDTH = os.environ.get("WIDTH", "832")
HEIGHT = os.environ.get("HEIGHT", "480")
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "1800"))

_ports_env = os.environ.get("PORTS", "")
if _ports_env:
    PORTS = [int(p) for p in _ports_env.split()]
else:
    PORTS = [PORT_BASE + i for i in range(NUM_SERVICES)]

# --- required local inputs (case directory, mirroring run.sh's INPUT_DIR) ---

# The ONLY input knob is INPUT_DIR: each case is one directory holding
# prompt.txt plus 0-2 reference frame images (0 = text-only, 1 = first
# frame, 2 = first + last frame). All inputs are read from it — no
# PROMPT_FILE/FRAMES env overrides.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
INPUT_DIR = os.environ.get("INPUT_DIR", os.path.join(REPO_ROOT, "inputs", "i2va"))

PROMPT_FILE = os.path.join(INPUT_DIR, "prompt.txt")
if not os.path.isfile(PROMPT_FILE):
    sys.exit(f"ERROR: {PROMPT_FILE} not found (check INPUT_DIR)")
with open(PROMPT_FILE, encoding="utf-8") as fh:
    PROMPT = fh.read()

# Reference frames: INPUT_DIR images (everything except prompt.txt/README*)
# in sorted filename order — the upload order (first, then last).
try:
    _names = sorted(os.listdir(INPUT_DIR))
except OSError as exc:
    sys.exit(f"ERROR: cannot read INPUT_DIR {INPUT_DIR}: {exc}")
REF_FILES = [os.path.join(INPUT_DIR, n) for n in _names
               if n != "prompt.txt" and not n.startswith("README")]
print(f"[generate.py] prompt:  {PROMPT_FILE}")
print(f"[generate.py] frames:  {' '.join(REF_FILES) or '<none — text-only request>'}\n")
for path in REF_FILES:
    if not os.path.isfile(path):
        sys.exit(f"ERROR: reference frame not found: {path}")
if len(REF_FILES) > 2:
    sys.exit(f"ERROR: FRAMES accepts at most 2 files (first [+ last]), "
             f"got {len(REF_FILES)}")
REF_DESC = {
    0: "0 reference frames (text-only)",
    1: "first frame only",
    2: "first + last frame",
}[len(REF_FILES)]

EXTRA_PARAMS = json.dumps(
    {"task": TASK_TYPE, "duration": int(DURATION), "audio_flow_shift": 3.0}
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def post_one(port: int, out_path: str) -> dict:
    """Send one FL2VA request; returns a result record."""
    url = f"http://{HOST}:{port}/v1/videos/sync"
    form = {
        "prompt": PROMPT,
        "fps": "24",
        "num_inference_steps": "50",
        "flow_shift": "12",
        "seed": SEED,
        "width": WIDTH,
        "height": HEIGHT,
        "extra_params": EXTRA_PARAMS,
    }
    # One repeated "input_reference" file field per frame (order matters:
    # first, then last) — mirrors the bash client's FRAME_FLAGS.
    frame_handles = [open(p, "rb") for p in REF_FILES]
    files = [("input_reference", (os.path.basename(p), fh, "image/png"))
             for p, fh in zip(REF_FILES, frame_handles)]
    started = time.monotonic()
    try:
        resp = requests.post(url, data=form, files=files,
                             timeout=REQUEST_TIMEOUT)
        elapsed = time.monotonic() - started
        body = resp.content
        with open(out_path, "wb") as fh:
            fh.write(body)
        if resp.status_code == 200 and body:
            return {"ok": True, "code": resp.status_code,
                    "elapsed": elapsed, "error": ""}
        return {"ok": False, "code": resp.status_code,
                "elapsed": elapsed,
                "error": f"non-200 or empty body (see {out_path})"}
    except requests.RequestException as exc:
        elapsed = time.monotonic() - started
        reason = str(exc) or exc.__class__.__name__
        return {"ok": False, "code": 0, "elapsed": elapsed, "error": reason}
    finally:
        for fh in frame_handles:
            fh.close()


def ffprobe_summary(out_path: str) -> str:
    if shutil.which("ffprobe") is None:
        return ""
    try:
        proc = subprocess.run(
            ["ffprobe",
             "-v", "error",
             "-select_streams", "v:0",
             "-show_entries", "stream=width,height,nb_frames,r_frame_rate",
             "-of", "csv=p=0",
             out_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    ports_str = " ".join(str(p) for p in PORTS)
    print(f"Posting {WIDTH}x{HEIGHT}/{DURATION}s {TASK_TYPE} request "
          f"({REF_DESC}) to {len(PORTS)} service(s): {ports_str}, "
          f"{ROUNDS} round(s) (concurrent fan-out per round)...\n")

    fail = False
    with ThreadPoolExecutor(max_workers=len(PORTS)) as pool:
        for rnd in range(1, ROUNDS + 1):
            print(f"=== Round {rnd}/{ROUNDS} ===")
            jobs = []
            for i, port in enumerate(PORTS):
                out = os.path.join(
                    OUT_DIR,
                    f"{TASK_TYPE}_r{rnd}_svc{i}_port{port}_seed{SEED}.mp4",
                )
                print(f"  service {i} -> http://{HOST}:{port}  -> {out}")
                jobs.append((i, port, out, pool.submit(post_one, port, out)))

            print(f"\nWaiting for all {len(PORTS)} request(s) of round "
                  f"{rnd} to finish...")
            print(f"Round {rnd} results:")
            for i, port, out, fut in jobs:
                res = fut.result()
                if res["ok"]:
                    line = (f"  [OK]  r{rnd} svc{i} port={port} "
                            f"-> {out}  [{res['elapsed']:.1f}s client]")
                    meta = ffprobe_summary(out)
                    if meta:
                        line += f"  [{meta}]"
                    print(line)
                else:
                    print(f"  [FAIL] r{rnd} svc{i} port={port} "
                          f"http={res['code']} elapsed={res['elapsed']:.1f}s "
                          f"error={res['error']}")
                    fail = True
            print()

    if fail:
        print("Some requests FAILED.")
        return 1
    print(f"All done: {ROUNDS} rounds x {len(PORTS)} service(s). "
          f"Outputs in {OUT_DIR}/ ({TASK_TYPE}_r<R>_svc<N>_...)")
    print("Read steady-state e2e_total_ms from each service log from "
          "round ~3 onward (rounds 1-2 are warmup/settling).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
