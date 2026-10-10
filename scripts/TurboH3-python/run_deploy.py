#!/usr/bin/env python3
"""TurboH3 deploy-only runner — pipeline.py's config, service exec'd directly.

pipeline.py couples deployment and traffic into one measured run; this
script splits out just the service for interactive use: the config file
stays the pipeline's JSON (only its "deploy" section is read — the
"generate"/"warmup" sections are ignored). cfg.build_cmd() is launched
with os.execvpe, which REPLACES this process with `vllm serve` (no Popen
child to manage): the service owns this terminal — logs stream here,
Ctrl-C/SIGTERM reach it natively — and its exit code becomes ours. The
Turbo artifact is preloaded exactly as a pipeline run would:
deploy.lora_path + deploy.lora_backend reach the server at startup.

Usage (from the repo root):
    python scripts/TurboH3-python/run_deploy.py \
        --config scripts/TurboH3-python/configs/fl2va/rtx5090/config.json
    python scripts/TurboH3-python/run_deploy.py --config <same> --dry-run
"""

import argparse
import os
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "MiniMax-H3-python"))

from deploy import DeployConfig  # noqa: E402
from pipeline import _read_config_doc  # noqa: E402


def _load_deploy_config(config: str | None) -> DeployConfig:
    """Pipeline config file -> its deploy section (env fills the rest).

    Same loading path as PipelineConfig.from_config()'s deploy half: the
    doc is validated as a whole (so the pipeline config works verbatim),
    then only doc["deploy"] is handed to DeployConfig.from_config().
    """
    if not config:
        return DeployConfig.from_env()
    return DeployConfig.from_config(_read_config_doc(config).get("deploy", {}))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, metavar="FILE",
                        help="pipeline config file (JSON, same as pipeline.py); "
                             "only its \"deploy\" section is used, env knobs "
                             "fill the rest")
    args = parser.parse_args()

    try:
        cfg = _load_deploy_config(args.config)
    except (OSError, ValueError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    
    # Direct launch: execvpe replaces THIS process with the service (no
    # Popen child, no stop bookkeeping) — logs stream to this terminal and
    # Ctrl-C reaches vllm natively. On success nothing below runs; the
    # service's exit status becomes this script's.
    cmd = cfg.build_cmd()
    env = cfg.build_env()
    print(f"[deploy] CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
    print(f"[deploy] {' '.join(shlex.quote(c) for c in cmd)}")
    sys.stdout.flush()      # exec drops unflushed C-level buffers
    try:
        os.execvpe(cmd[0], cmd, env)
    except OSError as exc:   # e.g. `vllm` not on PATH
        print(f"[deploy] FATAL: cannot exec {cmd[0]!r}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
