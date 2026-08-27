#!/usr/bin/env python3
"""Fan-out the 768p FL2VA request to N services concurrently, R rounds.

Python port of scripts/MiniMax-H3/generate/generate_768p_fanout.sh (drop-in
replacement: same env knobs, same request shape, same output naming).

Env knobs (all optional, defaults match the bash version):
  PORT_BASE      base port for contiguous derivation        (9000)
  NUM_SERVICES   number of services                          (1)
  PORTS          explicit space-separated port list; overrides base/count
  HOST           service host                                 (localhost)
  OUT_DIR        output directory                             (./outputs)
  ROUNDS         rounds of concurrent fan-out                 (7)
  SEED           generation seed                              (0)
  DURATION       audio/video seconds in extra_params          (15)
  FIRST_FRAME    local keyframe path (skips the CDN download)
  KEYFRAME_URL   CDN keyframe to download when FIRST_FRAME unset
  REQUEST_TIMEOUT  per-request read timeout, seconds         (1800)

Notes:
- The request is multipart form -> POST http://HOST:PORT/v1/videos/sync,
  exactly mirroring the curl -F fields of the bash client.
- Per-request wall time is measured client-side (in addition to the
  server-side e2e_total_ms you can grep from service logs).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
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
ROUNDS = int(os.environ.get("ROUNDS", "4"))
SEED = os.environ.get("SEED", "0")
DURATION = os.environ.get("DURATION", "15")
FIRST_FRAME = os.environ.get("FIRST_FRAME", "")
KEYFRAME_URL = os.environ.get(
    "KEYFRAME_URL",
    "https://cdn.hailuoai.com/prod/hailuo_demo/testsets/H3_AA_I2VA/"
    "gallery/sr_v17_variants_seed42_43_20260724/inputs/"
    "4a3a90bf9100_KDmcbkhzYo5sjjxr9FqcVmWVnzb.png",
)
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "1800"))

_ports_env = os.environ.get("PORTS", "")
if _ports_env:
    PORTS = [int(p) for p in _ports_env.split()]
else:
    PORTS = [PORT_BASE + i for i in range(NUM_SERVICES)]

PROMPT = """For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] This is a live-action, cinematic shot with a shallow depth of field. The camera holds a perfectly static shot throughout the entire eight-second duration, capturing a cozy family gathering in a traditional Japanese dining room. The scene opens with a large, intricately patterned blue and white ceramic bowl of ramen in the immediate foreground, rendered in crisp, sharp focus. The bowl sits on a smooth, polished long wooden table. Inside the bowl, a rich, oily golden-brown broth surrounds yellow wavy noodles, topped with two thick, round slices of chashu pork featuring visible fat marbling and a distinct spiral meat pattern. A generous mound of freshly chopped, bright green scallions rests in the center, and a crisp, dark green rectangular sheet of nori seaweed is tucked into the right edge. To the left of the bowl, a pair of light brown wooden chopsticks rests horizontally on a small, dark rectangular chopstick rest, near a small cylindrical ceramic teacup with blue painted patterns. On the right side of the table, a spherical paper lantern with a ribbed bamboo frame sits on a black wooden base. In the background, a large family of seven is gathered around the table, initially appearing as a soft, blurred presence. Behind them, traditional Japanese sliding shoji screens with wooden lattice frames are open, revealing a bright outdoor scene with lush green trees. Early in the clip, the thick, white steam rising from the hot ramen broth immediately intensifies, billowing upwards in thick, swirling clouds that dance continuously above the bowl. As the clip progresses into the middle seconds, the camera maintains its static position while the focus begins a deliberate, smooth shift deeper into the room. The foreground ramen bowl, its vibrant ingredients, and the rising steam gradually soften into a hazy, out-of-focus blur. Simultaneously, the family members in the background come into sharp, detailed clarity. The heavy steam continues to rise from the foreground, creating a dynamic, translucent veil between the camera and the family. With the focus now firmly locked on the background, the vibrant family dinner comes alive. The man in the dark navy blue long-sleeved shirt on the left leans forward, his mouth moving animatedly in a silent exchange. The young girl in the crisp white short-sleeved t-shirt beside him smiles brightly, looking toward the center of the table. The woman on the far left, wearing a soft light blue long-sleeved blouse, turns her head slightly, smiling gently. Across the table, the woman in the light grey button-down shirt smiles broadly, her eyes crinkling, as she rests her hands near her plate. The woman in the dark grey top further back uses her wooden chopsticks to pick up a small piece of food from a central ceramic dish filled with bright red pickled vegetables. The woman in the center back in the light grey sweater smiles gently, her hands clasped softly in front of her, observing the interaction. Throughout the remainder of the clip, the family continues their lively physical interaction, their mouths moving in continuous, silent cadences of conversation, while the thick, white steam from the blurred ramen bowl in the foreground never stops rising, adding a comforting atmosphere to the warm gathering.

overall_soundscape: The soundscape begins with a quiet room tone mixed with the faint, airy rustle of the thick steam billowing from the hot ramen bowl in the foreground, accompanied by the subtle, continuous hissing and bubbling of the rich broth. As the visual focus shifts deeper into the room, the physical sounds of the bustling family dinner become dominant in the foreground. The clear, sharp clinking of ceramic bowls and wooden chopsticks touching plates is clearly heard as the family members reach for food. This is followed by the faint, muffled thud of a cup being set down on the smooth wooden table, and the subtle, rhythmic rustle of cotton and wool clothing as the family members lean forward and gesture, perfectly capturing the lively, physical atmosphere of the shared meal.

non_diegetic_music: A gentle, heartwarming acoustic guitar melody plays softly in the background, accompanied by the subtle, resonant notes of a traditional Japanese koto. The music maintains a slow, comforting tempo that enhances the cozy, nostalgic, and joyful atmosphere of the family gathering."""

EXTRA_PARAMS = json.dumps(
    {"task": "fl2va", "duration": int(DURATION), "audio_flow_shift": 3.0}
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def resolve_keyframe() -> tuple[str, bool]:
    """Return (path, created_temp). Downloads the CDN keyframe unless
    FIRST_FRAME points at an existing local file."""
    if FIRST_FRAME and os.path.isfile(FIRST_FRAME):
        return FIRST_FRAME, False
    path = tempfile.mktemp(suffix=".png")
    print(f"Downloading keyframe: {KEYFRAME_URL}")
    with requests.get(KEYFRAME_URL, timeout=120, stream=True) as resp:
        resp.raise_for_status()
        with open(path, "wb") as fh:
            shutil.copyfileobj(resp.raw, fh)
    return path, True


def post_one(port: int, out_path: str) -> dict:
    """Send one FL2VA request; returns a result record."""
    url = f"http://{HOST}:{port}/v1/videos/sync"
    keyframe = post_one._keyframe  # set by the caller (shared file handle path)
    form = {
        "prompt": PROMPT,
        "fps": "24",
        "num_inference_steps": "50",
        "flow_shift": "12",
        "seed": SEED,
        "short_edge": "768",
        "aspect_ratio": "auto",
        "extra_params": EXTRA_PARAMS,
    }
    started = time.monotonic()
    try:
        with open(keyframe, "rb") as fh:
            files = {"input_reference": ("keyframe.png", fh, "image/png")}
            resp = requests.post(
                url, data=form, files=files, timeout=REQUEST_TIMEOUT
            )
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
    keyframe, created_temp = resolve_keyframe()
    post_one._keyframe = keyframe
    try:
        ports_str = " ".join(str(p) for p in PORTS)
        print(f"Posting 768p/{DURATION}s FL2VA request to {len(PORTS)} "
              f"service(s): {ports_str}, {ROUNDS} round(s) "
              f"(concurrent fan-out per round)...\n")

        fail = False
        with ThreadPoolExecutor(max_workers=len(PORTS)) as pool:
            for rnd in range(1, ROUNDS + 1):
                print(f"=== Round {rnd}/{ROUNDS} ===")
                jobs = []
                for i, port in enumerate(PORTS):
                    out = os.path.join(
                        OUT_DIR,
                        f"fl2va_r{rnd}_svc{i}_port{port}_seed{SEED}.mp4",
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
              f"Outputs in {OUT_DIR}/ (fl2va_r<R>_svc<N>_...)")
        print("Read steady-state e2e_total_ms from each service log from "
              "round ~3 onward (rounds 1-2 are warmup/settling).")
        return 0
    finally:
        if created_temp:
            try:
                os.remove(keyframe)
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main())
