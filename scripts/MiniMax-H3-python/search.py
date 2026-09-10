#!/usr/bin/env python3
"""Experiment search over the single-trial pipeline (pipeline.py).

A search is just MANY runs: each point goes through Pipeline as its own
timestamped run directory under logs/ (config.json / deploy.log /
metrics.json / outputs/ — see pipeline.py). This module only expands the
points, drives the loop, and maintains a flat JSONL index pointing at the
run directories (append-per-trial, so an interrupted search keeps partial
results). The index is a derived cache: every run stays self-describing via
its config.json/metrics.json, so it can be rebuilt or ignored.

    from deploy import DeployConfig
    from generate import GenerateConfig
    from pipeline import PipelineConfig
    from search import SearchConfig, Search

    cfg = SearchConfig(
        pipeline_base=PipelineConfig(
            deploy_base=DeployConfig(cuda_visible_devices="4,5,6,7"),
            generate_base=GenerateConfig(duration=15),
        ),
        points=[
            {"tensor_parallel_size": 1},
            {"tensor_parallel_size": 2},
            {"tensor_parallel_size": 4, "quantization": "fp8"},
            {"cache_config.residual_diff_threshold": 0.06},
        ],
    )
    results = Search(cfg).run()          # one Pipeline run per point

Each point is ONE experiment: a {field: value} override dict applied over
pipeline_base (deploy and generate fields merged in one dict). Point keys
are dispatched by field name (deploy fields -> DeployConfig, generate
fields -> GenerateConfig), so a parameter is never declared twice; dict
fields are overridden via dotted paths that deep-merge (only the named
sub-key varies). port/ports and log/pid plumbing are owned by the pipeline
and cannot be overridden.

An explicit point list rather than a cartesian grid: real sweeps are
rarely full products — e.g. fp8 quantization, once it works, stays on in
every other experiment and is never multiplied into the remaining axes.
When you do want a product, expand it with itertools.product while
building the list.

CLI:
    python search.py [--config FILE] [--dry-run] [--limit N]
      env knobs: SEARCH_OUT (index JSONL, default logs/index.jsonl)
      --config: pipeline config file (same schema as pipeline.py --config /
      run snapshots; run_dir inside is ignored — every point gets its own
      fresh timestamped run_dir).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

from deploy import DeployConfig
from generate import GenerateConfig
from pipeline import Pipeline, PipelineConfig

# Grid axes may only touch these fields (DeployConfig / GenerateConfig field
# names); anything else is a typo and would silently no-op in replace().
_DEPLOY_FIELDS = {f.name for f in dataclasses.fields(DeployConfig)}
_GENERATE_FIELDS = {f.name for f in dataclasses.fields(GenerateConfig)}
# Fields the configs re-derive in __post_init__ — sweeping them would be
# silently discarded (same reason the from_config() constructors strip them).
_DEPLOY_DERIVED = DeployConfig.DERIVED_FIELDS
_GENERATE_DERIVED = GenerateConfig.DERIVED_FIELDS
_PIPELINE_OWNED = ("port", "ports", "log_path", "pid_file")


def _key_root(key: str) -> str:
    """Top-level field name of an override key ("a.b" -> "a")."""
    return key.split(".", 1)[0]


def _deep_set(base: dict, dotted_key: str, value) -> dict:
    """Copy of dict `base` with base[a][b]...=value for dotted_key."""
    out = dict(base)
    keys = dotted_key.split(".")
    node = out
    for key in keys[:-1]:
        node[key] = dict(node[key])   # copy each nesting level
        node = node[key]
    node[keys[-1]] = value
    return out


@dataclasses.dataclass
class SearchConfig:
    """One search: a shared PipelineConfig base + the experiment list.

    Each point is one experiment: a {field: value} override dict applied
    over pipeline_base (deploy and generate fields merged in one dict).
    Nested dict fields (e.g. DeployConfig.cache_config) are overridden via
    dotted paths ("cache_config.residual_diff_threshold") that deep-merge,
    so the rest of the dict keeps base values. An empty list means a
    single baseline run.
    """

    pipeline_base: PipelineConfig = dataclasses.field(
        default_factory=PipelineConfig)
    points: list[dict] = dataclasses.field(default_factory=list)
    index_path: str = "logs/index.jsonl"

    def __post_init__(self) -> None:
        for i, point in enumerate(self.points):
            where = f"points[{i}]"
            if not isinstance(point, dict):
                raise TypeError(f"{where} must be a dict of overrides, got "
                                f"{type(point).__name__}")
            for key in point:
                root = _key_root(key)
                if root not in _DEPLOY_FIELDS and root not in _GENERATE_FIELDS:
                    raise ValueError(
                        f"{where} key {key!r} is neither a DeployConfig nor "
                        f"a GenerateConfig field")
                if root in _PIPELINE_OWNED:
                    raise ValueError(
                        f"{where} key {key!r} is owned by the pipeline "
                        f"(per-run port/log plumbing); do not override it")
                if root in _DEPLOY_DERIVED:
                    raise ValueError(
                        f"{where} key {key!r} is a derived DeployConfig "
                        f"field (recomputed from devices/TP); set "
                        f"tensor_parallel_size instead")
                if root in _GENERATE_DERIVED:
                    raise ValueError(
                        f"{where} key {key!r} is a derived GenerateConfig "
                        f"field (recomputed from input_dir); set input_dir "
                        f"instead")

    def _overrides(self, point: dict, base, fields: set[str]) -> dict:
        """Constructor overrides for one config class from a merged point.

        Plain keys pass through; dotted keys ("cache_config.x") deep-merge
        into the base's current dict for that field.
        """
        overrides, dotted = {}, []
        for key, value in point.items():
            if _key_root(key) not in fields:
                continue
            if "." in key:
                dotted.append((key, value))
            else:
                overrides[key] = value
        for key, value in dotted:
            top = key.split(".", 1)[0]
            current = overrides.get(top, getattr(base, top))
            if not isinstance(current, dict):
                raise ValueError(
                    f"point key {top!r} is not a dict field; dotted paths "
                    f"only apply to dict fields like cache_config")
            overrides[top] = _deep_set(current, key.split(".", 1)[1], value)
        return overrides

    def expanded(self) -> list[tuple[dict, PipelineConfig]]:
        """(point, PipelineConfig) per experiment, each with a fresh run_dir.

        An empty points list means one baseline run ([{}]). Configs are
        built with run_dir=None (lazy): the timestamped directory is
        claimed per point by Pipeline.run()/ensure_run_dir() when that
        run actually starts, so expanding never burns directory names.
        replace() also re-derives the deploy parallel fields (usp /
        encoder TP / VAE patch are init=False) — a point overriding
        tensor_parallel_size moves them automatically.
        """
        pairs = []
        for point in self.points or [{}]:
            pairs.append((point, dataclasses.replace(
                self.pipeline_base, run_dir=None,
                deploy_base=dataclasses.replace(
                    self.pipeline_base.deploy_base,
                    **self._overrides(point, self.pipeline_base.deploy_base,
                                      _DEPLOY_FIELDS)),
                generate_base=dataclasses.replace(
                    self.pipeline_base.generate_base,
                    **self._overrides(point,
                                      self.pipeline_base.generate_base,
                                      _GENERATE_FIELDS)),
            )))
        return pairs

    def pipeline_configs(self) -> list[PipelineConfig]:
        return [cfg for _, cfg in self.expanded()]


class Search:
    """Drives one Pipeline run per point; appends to the flat index."""

    def __init__(self, cfg: SearchConfig):
        self.cfg = cfg
        self.results: list[dict] = []

    def run(self, limit: int | None = None) -> list[dict]:
        trials = self.cfg.expanded()
        if limit:
            trials = trials[:limit]
        total = len(trials)
        print(f"[search] {total} run(s); index -> {self.cfg.index_path}\n")
        os.makedirs(os.path.dirname(self.cfg.index_path) or ".",
                    exist_ok=True)
        for i, (point, pcfg) in enumerate(trials, 1):
            pcfg.ensure_run_dir()   # claim the timestamped dir for this run
            print(f"[search] === run {i}/{total}: {point_name(point) or '(baseline)'}"
                  f" -> {pcfg.run_dir} ===")
            row = Pipeline(pcfg).run()
            row["point"] = point
            self.results.append(row)
            with open(self.cfg.index_path, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            summary = (row["metrics"].get("e2e_total_ms", {}).get("steady_mean")
                       if row.get("ok") else row.get("error", "FAILED"))
            print(f"[search] run {i}/{total} done: ok={row['ok']} "
                  f"e2e_steady_mean_ms={summary}\n")
        return self.results


def point_name(point: dict) -> str:
    return "_".join(f"{k}={v}" for k, v in sorted(point.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, metavar="FILE",
                        help="pipeline config file (same schema as "
                             "pipeline.py --config); env knobs fill "
                             "anything it omits")
    parser.add_argument("--dry-run", action="store_true",
                        help="list the runs that would execute and exit")
    parser.add_argument("--limit", type=int, default=None,
                        help="run only the first N points")
    args = parser.parse_args()

    try:
        # Same construction story as pipeline.py main(): --config layers a
        # JSON file over the env-resolved PipelineConfig. A run_dir inside
        # the file is harmless — expanded() re-derives a fresh timestamped
        # run_dir for every point.
        pipeline_base = (PipelineConfig.from_config(args.config)
                         if args.config else PipelineConfig(
                             deploy_base=DeployConfig.from_env(),
                             generate_base=GenerateConfig.from_env()))
        cfg = SearchConfig(
            pipeline_base=pipeline_base,
            index_path=os.environ.get("SEARCH_OUT", "logs/index.jsonl"),
            # Experiments live here: each point is one run's overrides over
            # pipeline_base, e.g. {"tensor_parallel_size": 2,
            # "quantization": "fp8"} (dotted paths like cache_config.x
            # deep-merge into dict fields). Empty = single baseline run.
            points=[],
        )
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        for i, (point, pcfg) in enumerate(cfg.expanded(), 1):
            print(f"  run {i}: {point_name(point) or '(baseline)'}")
        print(f"{len(cfg.expanded())} run(s)")
        return 0
    results = Search(cfg).run(limit=args.limit)
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
