#!/usr/bin/env python3
"""TurboH3 config -> two shell scripts: the pipeline's deploy/generate parts.

Given a pipeline config file (the same JSON pipeline.py / run_deploy.py
take), export the run as plain bash so it can be read, tweaked and run
without the python wrapper:

    deploy.sh    the deploy section: build_env()'s env exports + the
                 PROFILER conditional + build_cmd()'s `vllm serve` command
                 (flag per line) — run it foreground, Ctrl-C stops it.
    generate.sh  the generate section: GenerateConfig.build_cmd()'s curl
                 (the request surface straight from build_form(), incl.
                 the Turbo artifacts' request-level "lora" activation),
                 one -F flag per line with the prompt kept in a $PROMPT
                 variable; status sidecar + ffprobe summary.

Usage (from the repo root):
    python scripts/TurboH3-python/export_serve.py \
        --config scripts/TurboH3-python/configs/fl2va/rtx5090/config.json
    # writes deploy.sh + generate.sh into --out_dir (default ./export)
"""

import argparse
import os
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "MiniMax-H3-python"))

from deploy import DeployConfig  # noqa: E402
from generate import GenerateConfig  # noqa: E402
from pipeline import _read_config_doc  # noqa: E402


def _format_serve_cmd(cmd: list[str]) -> str:
    """build_cmd() token list -> multi-line `vllm serve ... \\ ...` text."""
    lines = [f"vllm serve {shlex.quote(cmd[2])}"]
    i = 3
    while i < len(cmd):
        tok = cmd[i]
        if i + 1 < len(cmd) and not cmd[i + 1].startswith("--"):
            lines.append(f"  {tok} {shlex.quote(cmd[i + 1])}")
            i += 2
        else:
            lines.append(f"  {tok}")
            i += 1
    return " \\\n".join(lines)


def _deploy_sh(cfg: DeployConfig, config_path: str) -> str:
    env = cfg.build_env()
    return f"""#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}

export VLLM_WORKER_MULTIPROC_METHOD={env['VLLM_WORKER_MULTIPROC_METHOD']}
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT={env['VLLM_OMNI_VIDEO_SYNC_TIMEOUT']}
export PYTORCH_CUDA_ALLOC_CONF={env['PYTORCH_CUDA_ALLOC_CONF']}

{_format_serve_cmd(cfg.build_cmd())}
"""


def _generate_sh(gen: dict, port: int, config_path: str) -> str:
    """generate section -> single-request bash script, via build_cmd().

    The curl invocation comes from GenerateConfig.build_cmd() (single
    source of truth for the CLI view of the request; its -F fields come
    straight from build_form(), so export and driver cannot drift),
    reformatted one flag per line. Only the baked prompt text is swapped
    for a $PROMPT variable read from the case file, keeping the exported
    script readable.
    """
    cfg = GenerateConfig.from_config(gen)
    # cmd: curl -X POST <url> -F k=v ... (no -o; the script adds its own
    # out file + status sidecar).
    cmd = cfg.build_cmd(port=port, out_path=None)
    flag_lines = []
    for i in range(4, len(cmd), 2):
        value = cmd[i + 1]
        if cmd[i] == "-F" and value.startswith("prompt="):
            flag_lines.append('  -F "prompt=${PROMPT}"')
        else:
            flag_lines.append(f"  {cmd[i]} {shlex.quote(value)}")
    body = " \\\n".join(flag_lines)
    out_name = f"{cfg.task}_seed{cfg.seed}.mp4"
    return f"""#!/usr/bin/env bash
{cmd[0]} -sS {cmd[1]} {cmd[2]} {shlex.quote(cmd[3])} \\
{body}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, metavar="FILE",
                        help="pipeline config file (JSON), same as pipeline.py")
    parser.add_argument("--out_dir", default="./export", metavar="DIR",
                        help="directory to write deploy.sh / generate.sh "
                             "into (default: ./export)")
    args = parser.parse_args()

    try:
        doc = _read_config_doc(args.config)
        cfg = DeployConfig.from_config(doc.get("deploy", {}))
    except (OSError, ValueError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    gen = doc.get("generate", {})
    deploy_sh = _deploy_sh(cfg, args.config)
    generate_sh = _generate_sh(gen, cfg.port, args.config)

    os.makedirs(args.out_dir, exist_ok=True)
    for name, text in (("deploy.sh", deploy_sh), ("generate.sh", generate_sh)):
        path = os.path.join(args.out_dir, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.chmod(path, 0o755)
        print(f"[export] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
