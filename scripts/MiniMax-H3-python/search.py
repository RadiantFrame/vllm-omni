#!/usr/bin/env python3
"""Grid search over the single-trial pipeline (pipeline.py).

A search is just MANY runs: each grid point goes through Pipeline as its own
timestamped run directory under logs/ (config.json / deploy.log /
metrics.json / outputs/ — see pipeline.py). This module only expands the
grid, drives the loop, and maintains a flat JSONL index pointing at the run
directories (append-per-trial, so an interrupted search keeps partial
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
        grid={
            "usp": [1, 2],
            "cache_config.residual_diff_threshold": [0.04, 0.06],
        },
    )
    results = Search(cfg).run()          # one Pipeline run per grid point

Grid axes are dispatched by field name (deploy fields -> DeployConfig,
generate fields -> GenerateConfig), so a parameter is never declared twice;
dict fields are swept via dotted paths that deep-merge (only the named
sub-key varies). port/ports and log/pid plumbing are owned by the pipeline
and cannot be swept.

CLI:
    python search.py [--dry-run] [--limit N]
      env knobs: SEARCH_OUT (index JSONL, default logs/index.jsonl)
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
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
_PIPELINE_OWNED = ("port", "ports", "log_path", "pid_file")


def _axis_root(axis: str) -> str:
    """Top-level field name of an axis ("a.b" -> "a")."""
    return axis.split(".", 1)[0]


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
    """One search: a shared PipelineConfig base + the axes that vary.

    Axes are field names; nested dict fields (e.g. DeployConfig.cache_config)
    are swept via dotted paths: "cache_config.residual_diff_threshold".
    Dotted overrides deep-merge, so the rest of the dict keeps base values.
    """

    pipeline_base: PipelineConfig = dataclasses.field(
        default_factory=PipelineConfig)
    grid: dict[str, list] = dataclasses.field(default_factory=dict)
    index_path: str = "logs/index.jsonl"

    def __post_init__(self) -> None:
        for axis in self.grid:
            root = _axis_root(axis)
            if root not in _DEPLOY_FIELDS and root not in _GENERATE_FIELDS:
                raise ValueError(
                    f"grid axis {axis!r} is neither a DeployConfig nor a "
                    f"GenerateConfig field")
            if root in _PIPELINE_OWNED:
                raise ValueError(
                    f"grid axis {axis!r} is owned by the pipeline (per-run "
                    f"port/log plumbing); do not sweep it")

    def points(self) -> list[dict]:
        """All grid points as {field: value} dicts (deploy+generate merged)."""
        if not self.grid:
            return [{}]
        keys = list(self.grid)
        return [dict(zip(keys, combo))
                for combo in itertools.product(*(self.grid[k] for k in keys))]

    def _overrides(self, point: dict, base, fields: set[str]) -> dict:
        """Constructor overrides for one config class from a merged point.

        Plain keys pass through; dotted keys ("cache_config.x") deep-merge
        into the base's current dict for that field.
        """
        overrides, dotted = {}, []
        for key, value in point.items():
            if _axis_root(key) not in fields:
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
                    f"grid axis {top!r} is not a dict field; dotted paths "
                    f"only apply to dict fields like cache_config")
            overrides[top] = _deep_set(current, key.split(".", 1)[1], value)
        return overrides

    def expanded(self) -> list[tuple[dict, PipelineConfig]]:
        """(point, PipelineConfig) per grid point, each with a fresh run_dir.

        run_dir="" re-derives the timestamped directory via __post_init__,
        so every point becomes an independent run under logs/.
        """
        pairs = []
        for point in self.points():
            pairs.append((point, dataclasses.replace(
                self.pipeline_base, run_dir="",
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
    """Drives one Pipeline run per grid point; appends to the flat index."""

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
    parser.add_argument("--dry-run", action="store_true",
                        help="list the runs that would execute and exit")
    parser.add_argument("--limit", type=int, default=None,
                        help="run only the first N points")
    args = parser.parse_args()

    try:
        cfg = SearchConfig(
            pipeline_base=PipelineConfig(
                deploy_base=DeployConfig.from_env(),
                generate_base=GenerateConfig.from_env(),
            ),
            index_path=os.environ.get("SEARCH_OUT", "logs/index.jsonl"),
        )
        # CLI mode runs a single baseline run; to sweep axes, construct
        # SearchConfig in Python (see the module docstring) or add a grid
        # here.
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
