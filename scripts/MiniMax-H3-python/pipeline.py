#!/usr/bin/env python3
"""Single-trial pipeline: deploy -> generate (N rounds) -> parse metrics.

Composes deploy.py's Deployer with generate.py's Generator into one run:

    from deploy import DeployConfig
    from generate import GenerateConfig
    from pipeline import PipelineConfig, Pipeline

    cfg = PipelineConfig(
        deploy_base=DeployConfig(cuda_visible_devices="4,5,6,7"),
        generate_base=GenerateConfig(duration=15),
    )
    row = Pipeline(cfg).run()      # dict: ok / client metrics / log metrics

Every run owns one timestamped directory under logs/ holding ALL of its
artifacts — nothing leaks to global paths and runs never overwrite each
other:

    logs/20260904-153012/
      config.json     effective-config snapshot + meta (started/git/host)
      deploy.log      the Deployer service log
      metrics.json    parse_log summary for this run
      outputs/*.mp4   generated videos

The directory name carries only WHEN (timestamp); parameters live in
config.json — treat logs/ as a queryable library of runs (grid search is
just many runs; see search.py, which adds a flat JSONL index over them).

CLI (one run from env knobs):
    python pipeline.py
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import socket
import subprocess
import sys
import time

import parse_log
from deploy import DeployConfig, Deployer
from generate import GenerateConfig, Generator

LOGS_ROOT = "logs"


@dataclasses.dataclass
class PipelineConfig:
    """One run: base configs + the run directory that groups its artifacts.

    deploy_base / generate_base carry the whole request+deployment shape
    (GPU list, model, rounds, duration, input dir, ...); the derived
    deploy_cfg()/generate_cfg() only pin the per-run plumbing (log path,
    output dir, port alignment) inside run_dir.
    """

    deploy_base: DeployConfig = dataclasses.field(default_factory=DeployConfig)
    generate_base: GenerateConfig = dataclasses.field(
        default_factory=GenerateConfig)
    run_dir: str = ""    # derived: logs/<YYYYmmdd-HHMMSS> unless set
    warmup: int = 2      # leading requests parse_log drops from the log

    def __post_init__(self) -> None:
        if not self.run_dir:
            self.run_dir = _new_run_dir(LOGS_ROOT)

    def deploy_cfg(self) -> DeployConfig:
        """deploy_base with the service log inside run_dir."""
        return dataclasses.replace(
            self.deploy_base,
            log_path=os.path.join(self.run_dir, "deploy.log"),
        )

    def generate_cfg(self, port: int) -> GenerateConfig:
        """generate_base retargeted at this run's port and output dir.

        replace() re-runs GenerateConfig.__init__ (derived fields included),
        so generate_base's fixed knobs (rounds, duration, input_dir, ...)
        carry over and only the port / out_dir change.
        """
        return dataclasses.replace(
            self.generate_base, ports=[port],
            out_dir=os.path.join(self.run_dir, "outputs"))

    def snapshot(self) -> dict:
        """Effective-config snapshot for run_dir/config.json.

        Includes defaults and derived values so the run is reproducible from
        this file alone; the raw prompt text is omitted (input_dir +
        ref_files already identify it).
        """
        deploy = dataclasses.asdict(self.deploy_cfg())
        generate = dataclasses.asdict(self.generate_cfg(port=self.deploy_base.port))
        generate.pop("prompt", None)
        return {
            "meta": {
                "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                "hostname": socket.gethostname(),
                "git_commit": _git_commit(),
                "argv": sys.argv,
                "warmup": self.warmup,
                "gpus": _gpu_info(),
            },
            "deploy": deploy,
            "generate": generate,
        }


class Pipeline:
    """Runs one trial; records its artifacts under cfg.run_dir."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.row: dict | None = None

    def run(self) -> dict:
        """Deployer up -> Generator traffic -> stop -> parse the log.

        Returns the metrics row (also written to run_dir/metrics.json):
        {run_dir, ok, client_ok, client_elapsed_s, metrics}.
        """
        cfg = self.cfg
        os.makedirs(cfg.run_dir, exist_ok=True)
        with open(os.path.join(cfg.run_dir, "config.json"), "w") as fh:
            json.dump(cfg.snapshot(), fh, indent=2)

        self.row = {"run_dir": cfg.run_dir,
                    "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        deploy_cfg = cfg.deploy_cfg()
        try:
            with Deployer(deploy_cfg) as deployer:
                gen = Generator(cfg.generate_cfg(deployer.port))
                traffic_ok = gen.run()
                self.row["client_ok"] = traffic_ok
                self.row["client_elapsed_s"] = round(sum(
                    r["elapsed"] for r in gen.results), 1)
            # The service is stopped; its log is complete — parse it.
            self.row["metrics"] = summarize_log(deploy_cfg.log_path,
                                                cfg.warmup)
            self.row["ok"] = bool(traffic_ok
                                  and not self.row["metrics"]["failures"])
        except Exception as exc:  # deployment failure or parser crash
            self.row["ok"] = False
            self.row["error"] = f"{type(exc).__name__}: {exc}"
            self.row["metrics"] = {"failures": 1}

        with open(os.path.join(cfg.run_dir, "metrics.json"), "w") as fh:
            json.dump(self.row, fh, indent=2)
        return self.row


def summarize_log(path: str, warmup: int) -> dict:
    return parse_log.summarize(parse_log.parse_log(path), warmup)


# Run dirs issued by this process (same-second collisions are not yet on
# disk when PipelineConfig objects are built back-to-back, e.g. by search).
_ISSUED_RUN_DIRS: set[str] = set()


def _new_run_dir(root: str) -> str:
    """logs/<YYYYmmdd-HHMMSS>, with a -N suffix on same-second collisions."""
    base = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(root, base)
    n = 2
    while run_dir in _ISSUED_RUN_DIRS or os.path.exists(run_dir):
        run_dir = os.path.join(root, f"{base}-{n}")
        n += 1
    _ISSUED_RUN_DIRS.add(run_dir)
    return run_dir


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:
        return ""


def _gpu_info() -> dict[str, int]:
    """The host's full GPU inventory as {model: count}, best-effort.

    Covers every card on the machine (not just cuda_visible_devices — which
    GPUs a run used lives in deploy.cuda_visible_devices); mixed models just
    add up as separate entries. Returns {} without nvidia-smi.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
    except Exception:
        return {}
    counts: dict[str, int] = {}
    for line in out.splitlines():
        name = line.strip()
        if name:
            counts[name] = counts.get(name, 0) + 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", default=None,
                        help="override the auto-generated logs/<timestamp> dir")
    args = parser.parse_args()
    try:
        cfg = PipelineConfig(
            deploy_base=DeployConfig.from_env(),
            generate_base=GenerateConfig.from_env(),
            run_dir=args.run_dir or "",
        )
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    row = Pipeline(cfg).run()
    return 0 if row["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
