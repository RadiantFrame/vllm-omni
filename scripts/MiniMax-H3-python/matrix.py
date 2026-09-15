#!/usr/bin/env python3
"""Combination matrix test: input_dir x short_edge x aspect_ratio.

Drives one Pipeline run per REASONABLE combination of the paprika cabc
case family, reusing search.py's points machinery. Two deployment batches
(each deploys its own model per run, like every pipeline run):

  batch "fl2va"  (FL2VA model):  fl2va points + t2va points
  batch "ref2va" (Ref2VA model): ref2va points

Combinations (durations 15/10/5 s by default, context-IR prompt on,
rounds=1; 10 combos x N durations runs total):

  ref2va  inputs/paprika-repro-task_cabc...          3 image refs
          aspect 16:9 / 9:16   x  short_edge 768 / 480   -> 4 runs
          (adaptive == 16:9 default, redundant -> skipped)
  fl2va   inputs/paprika-repro-task_cabc...-fl2va     1 image ref (3:2)
          aspect adaptive     x  short_edge 768 / 480   -> 2 runs
          (named ratios are advisory for fl2va — canvas always follows
          the first image — so they duplicate adaptive -> skipped)
  t2va    inputs/paprika-repro-task_cabc...-t2va      empty references/
          aspect 16:9 / 9:16   x  short_edge 768 / 480   -> 4 runs
          (adaptive is rejected for t2va by contract)

After the runs, every output video is ffprobe'd and compared against the
expected canvas (the official area policy mirrored locally; fl2va derives
its ratio from the first reference image):

  expected(short_edge, ratio) = align32 both axes after capping area at
  short_edge x align32(short_edge * 7/4)

Usage:
    python matrix.py --dry-run          # list points + expected dims, exit
    python matrix.py [--batch fl2va|ref2va] [--limit N]
                     [--durations 15,10,5]

Exit 0 iff every run succeeded AND every output resolution matches.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from pipeline import PipelineConfig
from search import Search, SearchConfig

CASES = Path("/data/jw/workspace/vllm-omni/inputs")
BASE = CASES / "paprika-repro-task_cabc382d8a081e6161a5"

# Mirrors _resolve_output_canvas / _align_multiple in the pipeline (verified
# against the official 768/480 ground truths: 1344x768, 768x1344, 480x832).
def _align32(v: float) -> int:
    return max(32, int(round(v / 32)) * 32)

def expected_canvas(short_edge: int, ratio: float) -> tuple[int, int]:
    """(width, height) the official area policy resolves to."""
    w, h = (short_edge * ratio, float(short_edge)) if ratio >= 1 else \
           (float(short_edge), short_edge / ratio)
    cap = short_edge * _align32(short_edge * 7 / 4)
    if w * h > cap:
        s = (cap / (w * h)) ** 0.5
        w, h = w * s, h * s
    return _align32(w), _align32(h)

def first_ref_ratio(input_dir: str) -> float:
    from PIL import Image
    refs = sorted((Path(input_dir) / "references").iterdir())
    with Image.open(refs[0]) as im:
        return im.size[0] / im.size[1]

def _points_for(task: str, input_dir: str, ratios: list[str],
                durations: list[int]) -> list[dict]:
    return [
        {"task": task, "input_dir": input_dir, "duration": dur,
         "use_context_ir_prompt": True,
         "short_edge": se, "aspect_ratio": ar}
        for dur in durations for se in (768, 480) for ar in ratios
    ]

def batches(durations: list[int]) -> dict[str, tuple[str, list[dict]]]:
    """batch name -> (preset path, points)."""
    fl2va_points = (
        _points_for("fl2va", f"{BASE}-fl2va", ["adaptive"], durations)
        + _points_for("t2va", f"{BASE}-t2va", ["16:9", "9:16"], durations)
    )
    ref2va_points = _points_for("ref2va", str(BASE), ["16:9", "9:16"],
                                durations)
    return {
        "fl2va": ("configs/fl2va/a100/config.json", fl2va_points),
        "ref2va": ("configs/ref2va/a100/config.json", ref2va_points),
    }

def expected_dims(point: dict) -> tuple[int, int]:
    ratio = (first_ref_ratio(point["input_dir"])
             if point["task"] == "fl2va"
             else {"16:9": 16 / 9, "9:16": 9 / 16}[point["aspect_ratio"]])
    return expected_canvas(point["short_edge"], ratio)

def point_label(point: dict) -> str:
    return (f"{point['task']}/{point['aspect_ratio']}@{point['short_edge']}p"
            f"_d{point['duration']}s"
            f"[{Path(point['input_dir']).name.split('-')[-1]}]")

def run_batch(name: str, preset: str, points: list[dict], limit: int | None):
    # Build the base from the preset but strip the generate section's
    # input_dir / use_context_ir_prompt: every point supplies its own, and
    # the preset's case dir may lack the IR prompt the preset enables
    # (e.g. r2va-omni) — the base's input would fail construction before
    # the point overrides ever apply.
    doc = json.load(open(preset))
    gen = {k: v for k, v in doc.get("generate", {}).items()
           if k not in ("input_dir", "use_context_ir_prompt")}
    from deploy import DeployConfig
    from generate import GenerateConfig
    base = PipelineConfig(
        deploy_base=DeployConfig.from_config(doc.get("deploy", {})),
        generate_base=GenerateConfig.from_config(gen),
    )
    base = replace(base, generate_base=replace(
        base.generate_base, rounds=1))          # dims verification: 1 round
    cfg = SearchConfig(pipeline_base=base, points=points)
    return Search(cfg).run(limit=limit)

def ffprobe_dims(mp4: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(mp4)],
        capture_output=True, text=True, timeout=60,
        check=True).stdout.strip()
    w, h = out.split(",")
    return int(w), int(h)

def verify(results: list[dict]) -> int:
    """Compare every run's output resolution against the expected canvas."""
    bad = 0
    print(f"\n{'point':<38}{'expected':>12}{'actual':>12}  verdict")
    print("-" * 78)
    for row in results:
        point = row["point"]
        label = point_label(point)
        exp = expected_dims(point)
        outs = sorted(Path(row["run_dir"], "outputs").glob("*.mp4"))
        if not row.get("ok") or not outs:
            print(f"{label:<38}{f'{exp[0]}x{exp[1]}':>12}{'NO OUTPUT':>12}  FAIL")
            bad += 1
            continue
        act = ffprobe_dims(outs[0])
        verdict = "OK" if act == exp else "MISMATCH"
        bad += verdict != "OK"
        print(f"{label:<38}{f'{exp[0]}x{exp[1]}':>12}"
              f"{f'{act[0]}x{act[1]}':>12}  {verdict}")
    return bad

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="list points with expected dims and exit")
    ap.add_argument("--batch", choices=["fl2va", "ref2va"], default=None,
                    help="run only one deployment batch")
    ap.add_argument("--limit", type=int, default=None,
                    help="run only the first N points per batch")
    ap.add_argument("--durations", default="15,10,5", metavar="LIST",
                    help="comma-separated target video seconds swept by "
                         "every combination (15,10,5)")
    args = ap.parse_args()

    try:
        durations = [int(d) for d in args.durations.split(",")]
    except ValueError:
        print(f"ERROR: --durations must be comma-separated ints, got "
              f"{args.durations!r}", file=sys.stderr)
        return 1
    if not durations or any(not 4 <= d <= 15 for d in durations):
        print(f"ERROR: durations must be in [4, 15], got {durations}",
              file=sys.stderr)
        return 1
    todo = batches(durations)
    if args.batch:
        todo = {args.batch: todo[args.batch]}

    if args.dry_run:
        for name, (preset, points) in todo.items():
            print(f"batch {name} (deploy: {preset})")
            for p in points:
                w, h = expected_dims(p)
                print(f"  {point_label(p):<40} -> {w}x{h}")
            print(f"  {len(points)} run(s)")
        return 0

    all_results: list[dict] = []
    for name, (preset, points) in todo.items():
        print(f"\n===== batch {name} (deploy: {preset}) =====")
        all_results += run_batch(name, preset, points, args.limit)

    bad = verify(all_results)
    print(f"\n{len(all_results) - bad}/{len(all_results)} combinations OK")
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
