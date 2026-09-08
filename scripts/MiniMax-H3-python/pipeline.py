#!/usr/bin/env python3
"""Single-trial pipeline: deploy -> generate (N rounds) -> collect metrics.

Composes deploy.py's Deployer, generate.py's Generator and metrics.py's
Metrics into one run:

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

CLI (one run from a config file, recommended for reproducible setups):
    python pipeline.py --config my_run.json

Precedence: dataclass defaults < env knobs < config file < --run-dir.
Each class owns its own construction: PipelineConfig.from_config() reads
the file and delegates the deploy/generate sections to
DeployConfig.from_config() / GenerateConfig.from_config(); keys unknown
to the target class are rejected (typos fail loudly instead of
silently no-op'ing).
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

from deploy import DeployConfig, Deployer
from generate import GenerateConfig, Generator
from metrics import Metrics

LOGS_ROOT = "logs"

_TOP_LEVEL_KEYS = {"deploy", "generate", "warmup", "run_dir", "meta"}


def _read_config_doc(source) -> dict:
    """Config source (JSON file path or dict) -> validated top-level dict.

    "meta" (present in run snapshots) is accepted but dropped — it is run
    bookkeeping, not configuration.
    """
    if isinstance(source, dict):
        doc, where = source, "config dict"
    else:
        try:
            with open(source, encoding="utf-8") as fh:
                doc = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source} is not valid JSON: {exc}") from exc
        except OSError as exc:
            raise ValueError(f"cannot read config file {source}: {exc}") from exc
        where = source
    if not isinstance(doc, dict):
        raise ValueError(f"{where}: top level must be a JSON object")
    unknown = set(doc) - _TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(f"{where}: unknown top-level key(s) {sorted(unknown)}; "
                         f"valid: {sorted(_TOP_LEVEL_KEYS - {'meta'})}")
    doc.pop("meta", None)
    return doc


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
    warmup: int = 2      # leading requests Metrics drops from the log

    @classmethod
    def from_config(cls, source, run_dir: str = "") -> "PipelineConfig":
        """Config-file constructor, symmetric to the parts' from_env().

        source: path to a JSON file (or an already-parsed dict). Shape —
        every key is optional; a section's keys override env knobs:

            {
              "deploy":   {"cuda_visible_devices": "4,5,6,7", "usp": 4, ...},
              "generate": {"task_type": "ref2va", "duration": 15, ...},
              "warmup":   2,
              "run_dir":  "logs/my-fixed-name"
            }

        deploy/generate are validated and applied by DeployConfig.from_config
        / GenerateConfig.from_config (unknown keys fail loudly; partial
        deploy.cache_config merges over the defaults). Precedence overall:
        dataclass defaults < env knobs < config file < run_dir argument.
        """
        doc = _read_config_doc(source)
        cfg = cls(
            deploy_base=DeployConfig.from_config(doc.get("deploy", {})),
            generate_base=GenerateConfig.from_config(doc.get("generate", {})),
            run_dir=run_dir or doc.get("run_dir", ""),
        )
        if "warmup" in doc:
            if not isinstance(doc["warmup"], int):
                raise ValueError(f"'warmup' must be an int, got "
                                 f"{type(doc['warmup']).__name__}")
            cfg.warmup = doc["warmup"]
        return cfg

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

        Schema mirrors the --config file (deploy/generate/warmup sections)
        plus a read-only "meta" block that from_config() skips — so the
        snapshot can be fed straight back as --config to replay a run
        (run_dir is omitted: a replay gets a fresh timestamped dir).
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
                "gpus": _gpu_info(),
            },
            "deploy": deploy,
            "generate": generate,
            "warmup": self.warmup,
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
            # The service is stopped; its log is complete — collect metrics.
            self.row["metrics"] = Metrics(warmup=cfg.warmup).collect(
                deploy_cfg.log_path)
            self.row["ok"] = bool(traffic_ok
                                  and not self.row["metrics"]["failures"])
        except Exception as exc:  # deployment failure or metrics crash
            self.row["ok"] = False
            self.row["error"] = f"{type(exc).__name__}: {exc}"
            self.row["metrics"] = {"failures": 1}

        with open(os.path.join(cfg.run_dir, "metrics.json"), "w") as fh:
            json.dump(self.row, fh, indent=2)
        return self.row


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
    parser.add_argument("--config", default=None, metavar="FILE",
                        help="pipeline config file (JSON; sections: deploy, "
                             "generate, warmup, run_dir). File keys override "
                             "env knobs; --run-dir overrides the file")
    parser.add_argument("--run-dir", default=None,
                        help="override the auto-generated logs/<timestamp> dir")
    args = parser.parse_args()
    try:
        if args.config:
            cfg = PipelineConfig.from_config(args.config,
                                             run_dir=args.run_dir or "")
        else:
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
