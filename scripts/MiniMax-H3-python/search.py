#!/usr/bin/env python3
"""Grid search over the single-trial pipeline (pipeline.py).

Expands a grid of DeployConfig/GenerateConfig field axes into one
PipelineConfig per point, runs each through Pipeline, and accumulates the
metric rows into a JSONL file (append-per-trial, so an interrupted search
keeps partial results):

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
            "residual_diff_threshold": [0.04, 0.06],
        },
    )
    results = Search(cfg).run()          # one Pipeline trial per grid point

Grid axes are dispatched by field name (deploy fields -> DeployConfig,
generate fields -> GenerateConfig), so a parameter is never declared twice;
port/ports and log/pid plumbing are owned by the pipeline and cannot be
swept.

CLI:
    python search.py [--dry-run] [--limit N]
      env knobs: SEARCH_OUT (results JSONL, default logs/search/results.jsonl)
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


@dataclasses.dataclass
class SearchConfig:
    """One search: a shared PipelineConfig base + the axes that vary."""

    pipeline_base: PipelineConfig = dataclasses.field(
        default_factory=PipelineConfig)
    grid: dict[str, list] = dataclasses.field(default_factory=dict)
    results_path: str = "logs/search/results.jsonl"

    def __post_init__(self) -> None:
        for axis in self.grid:
            if axis not in _DEPLOY_FIELDS and axis not in _GENERATE_FIELDS:
                raise ValueError(
                    f"grid axis {axis!r} is neither a DeployConfig nor a "
                    f"GenerateConfig field")
            if axis in ("port", "ports", "log_path", "pid_file"):
                raise ValueError(
                    f"grid axis {axis!r} is owned by the pipeline (per-trial "
                    f"port/log plumbing); do not sweep it")

    def points(self) -> list[dict]:
        """All grid points as {field: value} dicts (deploy+generate merged)."""
        if not self.grid:
            return [{}]
        keys = list(self.grid)
        return [dict(zip(keys, combo))
                for combo in itertools.product(*(self.grid[k] for k in keys))]

    def pipeline_configs(self) -> list[PipelineConfig]:
        """One PipelineConfig per grid point, named after its point."""
        cfgs = []
        for i, point in enumerate(self.points(), 1):
            deploy_overrides = {k: v for k, v in point.items()
                                if k in _DEPLOY_FIELDS}
            generate_overrides = {k: v for k, v in point.items()
                                  if k in _GENERATE_FIELDS}
            cfgs.append(dataclasses.replace(
                self.pipeline_base,
                name=point_name(point) or f"trial{i}",
                deploy_base=dataclasses.replace(self.pipeline_base.deploy_base,
                                                **deploy_overrides),
                generate_base=dataclasses.replace(
                    self.pipeline_base.generate_base,
                    **generate_overrides),
            ))
        return cfgs


class Search:
    """Drives one Pipeline trial per grid point; accumulates metric rows."""

    def __init__(self, cfg: SearchConfig):
        self.cfg = cfg
        self.results: list[dict] = []

    def run(self, limit: int | None = None) -> list[dict]:
        pipeline_cfgs = self.cfg.pipeline_configs()
        if limit:
            pipeline_cfgs = pipeline_cfgs[:limit]
        total = len(pipeline_cfgs)
        print(f"[search] {total} trial(s); results -> "
              f"{self.cfg.results_path}\n")
        os.makedirs(os.path.dirname(self.cfg.results_path) or ".",
                    exist_ok=True)
        for i, pcfg in enumerate(pipeline_cfgs, 1):
            print(f"[search] === trial {i}/{total}: {pcfg.name} ===")
            row = Pipeline(pcfg).run()
            self.results.append(row)
            with open(self.cfg.results_path, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            summary = (row["metrics"].get("e2e_total_ms", {}).get("steady_mean")
                       if row.get("ok") else row.get("error", "FAILED"))
            print(f"[search] trial {i}/{total} done: ok={row['ok']} "
                  f"e2e_steady_mean_ms={summary}\n")
        return self.results


def point_name(point: dict) -> str:
    return "_".join(f"{k}={v}" for k, v in sorted(point.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="list the trials that would run and exit")
    parser.add_argument("--limit", type=int, default=None,
                        help="run only the first N trials")
    args = parser.parse_args()

    try:
        cfg = SearchConfig(
            pipeline_base=PipelineConfig(
                deploy_base=DeployConfig.from_env(),
                generate_base=GenerateConfig.from_env(),
            ),
            results_path=os.environ.get("SEARCH_OUT",
                                        "logs/search/results.jsonl"),
        )
        # CLI mode runs a single baseline trial; to sweep axes, construct
        # SearchConfig in Python (see the module docstring) or add a grid
        # here.
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        for i, pcfg in enumerate(cfg.pipeline_configs(), 1):
            print(f"  trial {i}: {pcfg.name}")
        print(f"{len(cfg.pipeline_configs())} trial(s)")
        return 0
    results = Search(cfg).run(limit=args.limit)
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
