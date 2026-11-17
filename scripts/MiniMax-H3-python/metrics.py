#!/usr/bin/env python3
"""Extract metrics from a finished vLLM-Omni deploy/generate run.

The pipeline's third phase (deploy -> generate -> METRICS) lives here:
`Metrics` turns a run's artifacts into the summary dict written to
run_dir/metrics.json. Its current source is the service log via
`LogParser`:

  latency    per-request [OmniTiming] lines; the summary keys reuse the
             line's own field names (total/engine, seconds). Steady-state
             aggregation drops the leading warmup rounds (1-2 by default)
  cache-dit  executed/cached steps and residual-diff percentiles per request
  resources  model-load time/GiB, per-worker GPU memory after load
  failures   FATAL / Traceback / OOM / health-timeout occurrences

The per-request stats tables (RequestE2EStats / StageRequestStats) are
DEBUG-only upstream; this parser intentionally does NOT depend on them.
Anything they used to carry (denoise_step_latency_ms, postprocess_time_s,
shape_check) is not extractable from [OmniTiming] and was dropped —
validate run shape from config.json and ffprobe over the output mp4s
instead.

Usage:
  python metrics.py [LOG_PATH] [--warmup N] [--json]
  python metrics.py logs/deploy.log --json > metrics.json

The per-request series is keyed by request order of appearance in the log
(the generate.py rounds fire sequentially, so order == round number).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys

DEFAULT_LOG = "logs/deploy.log"

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# [OmniTiming] req=video_sync-... total=594.30s engine=594.30s stages=[0:594.30s]
OMNI_TIMING_RE = re.compile(
    r"\[OmniTiming\] req=\S+ total=([0-9.]+)s engine=([0-9.]+)s")

# [Cache-DiT] ... Cache Steps and Residual Diffs Statistics: <cls>,
#   Executed Steps: 30, Transformer Executed Steps: 50
CACHE_STATS_RE = re.compile(
    r"Cache Steps and Residual Diffs Statistics: (.*?), "
    r"Executed Steps: (\d+), Transformer Executed Steps: (\d+)")
# | 20          | 0.001 | ... |  (the row under the Cache Steps table header)
CACHE_DIFFS_ROW_RE = re.compile(
    r"\|\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|"
    r"\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|")

MODEL_LOAD_RE = re.compile(r"Model loading took ([0-9.]+) GiB and ([0-9.]+) seconds")
WORKER_MEM_RE = re.compile(
    r"Worker (\d+): Process-scoped GPU memory after model loading: ([0-9.]+) GiB")

FAILURE_PATTERNS = ("FATAL", "Traceback (most recent call last)",
                    "CUDA out of memory", "health check timed out",
                    "launcher exited")


class LogParser:
    """Parses one service log; summarize() extracts the metric summary.

    Typical use: LogParser(path).parse().summarize(warmup=2) — parse()
    returns self so the two steps chain.
    """

    def __init__(self, path: str):
        self.path = path
        self.omni_timings: list[tuple[float, float]] = []  # (total_s, engine_s)
        self.cache_dit: list[dict] = []
        self.model_load_gib: list[float] = []
        self.model_load_seconds: list[float] = []
        self.worker_gpu_mem_gib: dict[str, float] = {}
        self.failures: list[str] = []

    # -- parsing --------------------------------------------------------------

    def parse(self) -> "LogParser":
        """Read the log once and populate the instance fields."""
        with open(self.path, encoding="utf-8", errors="replace") as fh:
            lines = [ANSI_RE.sub("", ln.rstrip("\n")) for ln in fh]

        for ln in lines:
            m = CACHE_STATS_RE.search(ln)
            if m:
                self.cache_dit.append({
                    "module": m.group(1),
                    "executed_steps": int(m.group(2)),
                    "transformer_executed_steps": int(m.group(3)),
                })
                continue
            m = CACHE_DIFFS_ROW_RE.search(ln)
            if m and self.cache_dit:
                row = [float(g) for g in m.groups()[1:]]
                stats = self.cache_dit[-1]
                stats["cached_steps"] = int(m.group(1))
                for name, val in zip(("p00", "p25", "p50", "p75", "p95",
                                      "min", "max"), row):
                    stats[f"residual_{name}"] = val
                continue
            m = OMNI_TIMING_RE.search(ln)
            if m:
                self.omni_timings.append((float(m.group(1)), float(m.group(2))))
                continue
            m = MODEL_LOAD_RE.search(ln)
            if m:
                self.model_load_gib.append(float(m.group(1)))
                self.model_load_seconds.append(float(m.group(2)))
                continue
            m = WORKER_MEM_RE.search(ln)
            if m:
                self.worker_gpu_mem_gib[f"worker_{m.group(1)}"] = float(m.group(2))
                continue
            for pat in FAILURE_PATTERNS:
                if pat in ln:
                    self.failures.append(ln.strip())
                    break

        return self

    # -- summarizing ----------------------------------------------------------

    def summarize(self, warmup: int = 2) -> dict:
        """Steady-state metric summary (leading `warmup` requests dropped)."""
        e2e_all = [round(total, 2) for total, _ in self.omni_timings]
        steady = e2e_all[warmup:]
        out: dict = {
            "num_requests": len(self.omni_timings),
            "warmup_excluded": min(warmup, len(e2e_all)),
            "failures": len(self.failures),
            "failure_samples": self.failures[:5],
        }
        if steady:
            out["total"] = {
                "steady_mean": round(statistics.mean(steady), 2),
                "steady_median": round(statistics.median(steady), 2),
                "steady_min": round(min(steady), 2),
                "steady_max": round(max(steady), 2),
                "steady_count": len(steady),
                "all": e2e_all,
            }
        # [OmniTiming] engine = total - preprocess; steady-window average.
        engine_vals = [engine for _, engine in self.omni_timings[warmup:]]
        if engine_vals:
            out["engine"] = round(statistics.mean(engine_vals), 2)
        # Cache-DiT: average across the per-request summaries.
        if self.cache_dit:
            def avg(key: str) -> float | None:
                vals = [c[key] for c in self.cache_dit if key in c]
                return round(statistics.mean(vals), 3) if vals else None
            steps = [c["executed_steps"] for c in self.cache_dit
                     if "executed_steps" in c]
            total = [c["transformer_executed_steps"] for c in self.cache_dit
                     if "transformer_executed_steps" in c]
            out["cache_dit"] = {
                "executed_steps_avg":
                    round(statistics.mean(steps), 1) if steps else None,
                "transformer_steps_avg":
                    round(statistics.mean(total), 1) if total else None,
                "residual_p95_avg": avg("residual_p95"),
                "residual_max_avg": avg("residual_max"),
            }
        if self.model_load_gib:
            out["model_load"] = {
                "gib": max(self.model_load_gib),
                "seconds": max(self.model_load_seconds),
            }
        if self.worker_gpu_mem_gib:
            out["worker_gpu_mem_gib_max"] = max(
                self.worker_gpu_mem_gib.values())
        return out


class Metrics:
    """Stage-3 collector: a finished run's artifacts -> the metrics dict.

    Pipeline-facing interface for the deploy -> generate -> METRICS phases;
    LogParser (the service-log source) is an implementation detail this
    class composes. The instance carries the analysis knobs (`warmup` =
    leading requests dropped as compile/settling warmup); the run's
    artifacts are collect() arguments, not constructor state.

    Extension point: visual-quality metrics over the run's generated
    videos belong in collect() as a second source, e.g.
    Metrics(warmup=...).collect(log_path, outputs_dir=...) adding
    SSIM/PSNR keys.
    """

    def __init__(self, warmup: int = 2):
        self.warmup = warmup

    def collect(self, log_path: str) -> dict:
        """Summarize the run's service log; shape mirrors
        LogParser.summarize()."""
        return LogParser(log_path).parse().summarize(self.warmup)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log_path", nargs="?", default=DEFAULT_LOG,
                        help=f"service log to parse (default: {DEFAULT_LOG})")
    parser.add_argument("--warmup", type=int, default=2,
                        help="leading requests to exclude as warmup (default: 2)")
    parser.add_argument("--json", action="store_true",
                        help="emit the summary as JSON (for the grid-search driver)")
    args = parser.parse_args()

    try:
        summary = Metrics(warmup=args.warmup).collect(args.log_path)
    except OSError as exc:
        print(f"ERROR: cannot read {args.log_path}: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print(f"=== {args.log_path} ===")
    print(f"requests parsed: {summary['num_requests']} "
          f"(steady: {summary.get('total', {}).get('steady_count', 0)}, "
          f"warmup excluded: {summary['warmup_excluded']})")
    if "total" in summary:
        e = summary["total"]
        print(f"total steady: mean={e['steady_mean']}s "
              f"median={e['steady_median']}s min={e['steady_min']}s "
              f"max={e['steady_max']}s")
        print(f"  per-round: {e['all']}")
    if "engine" in summary:
        print(f"engine (steady avg): {summary['engine']}s")
    if "cache_dit" in summary:
        print(f"cache-dit: {summary['cache_dit']}")
    if "model_load" in summary:
        print(f"model load: {summary['model_load']} GiB / seconds")
    if summary.get("worker_gpu_mem_gib_max"):
        print(f"worker gpu mem max: {summary['worker_gpu_mem_gib_max']} GiB")
    if summary["failures"]:
        print(f"FAILURES: {summary['failures']}")
        for s in summary["failure_samples"]:
            print(f"  {s[:200]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
