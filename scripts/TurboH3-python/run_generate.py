#!/usr/bin/env python3
"""TurboH3 generate-only runner — pipeline.py's config, traffic via generate.py.

pipeline.py couples deployment and traffic into one measured run; this
script splits out just the traffic, pairing with a service that is already
up (e.g. started by run_deploy.py with the same config file): the config
stays the pipeline's JSON — the "generate" section shapes the request,
the "deploy" section contributes only its port (the retarget pipeline.py's
generate_cfg() applies; --port overrides). Traffic is ONE request
(multipart POST /v1/videos/sync, generate.py's request shape inlined from
Generator._post_one) — no rounds x ports fan-out, for quick interactive
pokes at a live service.
The request carries the Turbo artifact contract from the generate section
— steps / flow shifts / request-level "lora" activation — exactly as a
pipeline run would.

Usage (from the repo root, service already up):
    python scripts/TurboH3-python/run_generate.py \
        --config scripts/TurboH3-python/configs/fl2va/rtx5090/config.json
    python scripts/TurboH3-python/run_generate.py --config <same> --port 9000
"""

import argparse
import dataclasses
import mimetypes
import os
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "MiniMax-H3-python"))

from generate import GenerateConfig, Generator  # noqa: E402
from pipeline import _read_config_doc  # noqa: E402


def _load_generate_config(config: str | None,
                          port: int | None) -> GenerateConfig:
    """Pipeline config file -> generate section (env fills the rest).

    Same loading path as PipelineConfig.from_config()'s generate half, plus
    the retarget pipeline.py's generate_cfg() applies: the request is pinned
    at the deploy section's port (the one run_deploy.py serves). --port
    overrides both; without a config file this is generate.py's env-knob
    CLI (ports from PORT_BASE/NUM_SERVICES/PORTS).
    """
    if not config:
        cfg = GenerateConfig.from_env()
        return dataclasses.replace(cfg, ports=[port]) if port else cfg
    doc = _read_config_doc(config)
    cfg = GenerateConfig.from_config(doc.get("generate", {}))
    # Pipeline parity (PipelineConfig.__post_init__): deploy.lora_path
    # preloads the adapter (run_deploy.py), generate.lora_path activates it
    # per request — a mismatch would activate a second adapter against the
    # other artifact's sampling contract.
    deploy = doc.get("deploy", {})
    generate = doc.get("generate", {})
    if deploy.get("lora_path") and generate.get("lora_path") \
            and deploy["lora_path"] != generate["lora_path"]:
        raise ValueError(
            f"deploy.lora_path and generate.lora_path must name the same "
            f"artifact (deploy preloads it, the request activates it): "
            f"got {deploy['lora_path']!r} vs {generate['lora_path']!r}")
    if not port:
        port = deploy.get("port")
    return dataclasses.replace(cfg, ports=[port]) if port else cfg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, metavar="FILE",
                        help="pipeline config file (JSON, same as pipeline.py); "
                             "its \"generate\" section shapes the request, the "
                             "\"deploy\" section contributes the port")
    parser.add_argument("--port", type=int, default=None, metavar="N",
                        help="target this service port (overrides the config's "
                             "deploy.port)")
    parser.add_argument("--out_path", type=str, default="./outputs/video.mp4",
                        metavar="FILE",
                        help="write the video to this path (parent dir is "
                             "created); default: ./outputs/video.mp4")
    args = parser.parse_args()

    try:
        cfg = _load_generate_config(args.config, args.port)
    except (OSError, ValueError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Single shot: one request to the first port, bypassing Generator.run()'s
    # rounds x ports fan-out (and its warmup-round framing). Below is
    # generate.py's Generator._post_one inlined, posting once and writing
    # the response to out_path.
    port = cfg.ports[0]
    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    print(f"[generate] prompt: {cfg.prompt_file}")
    print(f"[generate] frames: "
          f"{' '.join(cfg.ref_files) or '<none — text-only request>'}")
    print(f"[generate] POST http://{cfg.host}:{port}/v1/videos/sync "
          f"({cfg.task}, {cfg.num_inference_steps} steps) -> {args.out_path}")

    url = f"http://{cfg.host}:{port}/v1/videos/sync"
    form = cfg.build_form()
    # Repeated file field per reference (upload order = <Picture/Video N>
    # numbering in the prompt). Field name and MIME follow the task:
    # fl2va/t2va send "input_reference" image frames; ref2va sends
    # "input_references" mixed image/video/audio files whose modality the
    # server detects from the MIME type.
    field = "input_references" if cfg.task == "ref2va" else "input_reference"
    frame_handles = [open(p, "rb") for p in cfg.ref_files]
    files = [(field, (os.path.basename(p), fh,
                      mimetypes.guess_type(p)[0] or "image/png"))
             for p, fh in zip(cfg.ref_files, frame_handles)]
    
    resp = requests.post(url, data=form, files=files,
                            timeout=cfg.request_timeout)
    body = resp.content
    status = resp.status_code == 200 and body
    if status:
        with open(args.out_path, "wb") as fh:
            fh.write(body)
        
        line = f"[generate] OK  {args.out_path}"
        meta = Generator._ffprobe_summary(args.out_path)
        if meta:
            line += f"  [{meta}]"
        print(line)
    else:
        print(f"[generate] FAIL http={resp.status_code}", file=sys.stderr)
    for fh in frame_handles:
        fh.close()
    if status:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
