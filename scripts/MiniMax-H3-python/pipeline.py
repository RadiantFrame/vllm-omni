#!/usr/bin/env python3
"""Single-trial pipeline: deploy -> generate (N rounds) -> parse metrics.

Composes deploy.py's Deployer with generate.py's Generator into one trial:

    from deploy import DeployConfig
    from generate import GenerateConfig
    from pipeline import PipelineConfig, Pipeline

    cfg = PipelineConfig(
        name="baseline",
        deploy_base=DeployConfig(cuda_visible_devices="4,5,6,7"),
        generate_base=GenerateConfig(duration=15),
    )
    row = Pipeline(cfg).run()      # dict: ok / client metrics / log metrics

The trial's service log goes to logs/<name>.log (unique per name, so trials
never overwrite each other); after the service stops, parse_log.py extracts
steady-state metrics from exactly that file. Grid search over many trials
lives in search.py, built on top of this Pipeline.

CLI (one baseline trial from env knobs):
    python pipeline.py [NAME]
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time

import parse_log
from deploy import DeployConfig, Deployer
from generate import GenerateConfig, Generator


@dataclasses.dataclass
class PipelineConfig:
    """One trial: base configs + trial identity.

    deploy_base / generate_base carry the whole request+deployment shape
    (GPU list, model, rounds, duration, input dir, ...); the derived
    deploy_cfg()/generate_cfg() only pin the per-trial plumbing (log path,
    port alignment) that the Pipeline owns.
    """

    deploy_base: DeployConfig = dataclasses.field(default_factory=DeployConfig)
    generate_base: GenerateConfig = dataclasses.field(
        default_factory=GenerateConfig)
    name: str = "trial"
    warmup: int = 2   # leading requests parse_log drops from the trial log

    def deploy_cfg(self) -> DeployConfig:
        """deploy_base with per-trial log/pid paths under logs/."""
        return dataclasses.replace(
            self.deploy_base,
            log_path=f"logs/{self.name}.log",
            pid_file=f"logs/{self.name}.pid",
        )

    def generate_cfg(self, port: int) -> GenerateConfig:
        """generate_base retargeted at this trial's service port.

        replace() re-runs GenerateConfig.__init__ (derived fields included),
        so generate_base's fixed knobs (rounds, duration, input_dir, ...)
        carry over and only the port changes.
        """
        return dataclasses.replace(self.generate_base, ports=[port])


class Pipeline:
    """Runs one trial; returns (and records) its metric row."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.row: dict | None = None

    def run(self) -> dict:
        """Deployer up -> Generator traffic -> stop -> parse the log.

        Returns the row: {name, ok, client_ok, client_elapsed_s, metrics}.
        """
        cfg = self.cfg
        self.row = {"name": cfg.name,
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
        return self.row


def summarize_log(path: str, warmup: int) -> dict:
    return parse_log.summarize(parse_log.parse_log(path), warmup)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", nargs="?", default="trial",
                        help="trial name (log goes to logs/<name>.log)")
    args = parser.parse_args()
    try:
        cfg = PipelineConfig(
            deploy_base=DeployConfig.from_env(),
            generate_base=GenerateConfig.from_env(),
            name=args.name,
        )
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    row = Pipeline(cfg).run()
    return 0 if row["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
