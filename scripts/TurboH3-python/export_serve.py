#!/usr/bin/env python3
"""TurboH3 config -> two shell scripts: the pipeline's deploy/generate parts.

Given a pipeline config file (the same JSON pipeline.py / run_deploy.py
take), export the run as plain bash so it can be read, tweaked and run
without the python wrapper. The rendering lives on the Config classes:

    DeployConfig.export_sh()    -> deploy.sh: the env exports + the
                                   `vllm serve` command, flag per line.
    GenerateConfig.export_sh()  -> generate.sh: build_cmd()'s curl, one
                                   -F flag per line, prompt in $PROMPT.

Usage (from the repo root):
    python scripts/TurboH3-python/export_serve.py \
        --config scripts/TurboH3-python/configs/fl2va/rtx5090/config.json
    # writes deploy.sh + generate.sh into --out_dir (default ./export)
"""

import argparse
import dataclasses
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "MiniMax-H3-python"))

from deploy import DeployConfig  # noqa: E402
from generate import GenerateConfig  # noqa: E402
from pipeline import _read_config_doc  # noqa: E402


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
        deploy_cfg = DeployConfig.from_config(doc.get("deploy", {}))
        gen_cfg = GenerateConfig.from_config(doc.get("generate", {}))
        # Pin the traffic at the service this config deploys.
        gen_cfg = dataclasses.replace(gen_cfg, ports=[deploy_cfg.port])
    except (OSError, ValueError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    for name, text in (("deploy.sh", deploy_cfg.export_sh()),
                       ("generate.sh", gen_cfg.export_sh())):
        path = os.path.join(args.out_dir, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.chmod(path, 0o755)
        print(f"[export] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
