#!/usr/bin/env python3
"""MiniMax-H3 FL2VA service deployer (Python port of the rtx5090 deploy.sh).

Two ways to use:

1. As a module (for pipeline.py / search.py):
     from deploy import DeployConfig, Deployer
     with Deployer(DeployConfig.from_env()) as d:   # serve + wait healthy
         gen = Generator.from_deployer(d)           # generate.py pairs with this
         gen.run()
     ...                                            # d stopped cleanly on exit
   Module level additionally exports graceful_stop_pid(pid) for stopping a
   service that is not owned by any Deployer here (e.g. a --detach leftover).

2. As a CLI (parity with deploy.sh):
     python deploy.py                 # foreground, Ctrl-C stops the service
     python deploy.py --detach        # background + log + wait healthy, then exit
                                      # (service keeps running; pid recorded in the
                                      # pid file, default logs/deploy.pid)
     python deploy.py --stop          # stop the --detach service via the pid file
     python deploy.py --stop-pid N    # stop an arbitrary pid the same way

Config knobs (env names = field names uppercased; defaults mirror 4rtx5090/deploy.sh):
  MODEL                      model path
  PORT                       service port                    (9000)
  CUDA_VISIBLE_DEVICES       gpu list                        (0,1,2,3)
  TENSOR_PARALLEL_SIZE       (4)
  USP                        (= GPU count / TP)
  RING                       (1)
  TEXT_ENCODER_TP_SIZE       (= GPU count)
  VAE_PATCH_PARALLEL_SIZE    (= GPU count)
  QUANTIZATION               "" (off) or "fp8"               ("")
  ENABLE_CPU_OFFLOAD         1/0                             (0)
  DIFFUSION_ATTENTION_BACKEND                                 (CUDNN_ATTN)
  CACHE_BACKEND    "" (off) or "cache_dit"                  ("")
  CACHE_CONFIG    JSON of cache-dit overrides merged over the defaults
                  (keys mirror vllm-omni's DiffusionCacheConfig), e.g.
                  '{"enable_taylorseer": true, "taylorseer_order": 2}'
  NUM_WEIGHT_LOAD_THREADS    (8)
  LOG / HEALTH_TIMEOUT_MIN   deployer-side knobs

Every deploy also exports: VLLM_WORKER_MULTIPROC_METHOD=spawn,
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=4500, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
(anti-fragmentation; 768p OOMed without it).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from typing import Any, Callable
import urllib.request
from dataclasses import dataclass, field

MODEL_DEFAULT = "/data/models/modelscope/MiniMax/MiniMax-H3/FL2VA"

# Cache-DiT keys accepted in cache_config: the official vllm-omni
# DiffusionCacheConfig set (the server's from_dict silently files unknown
# keys under _extra_params, so a typo would silently no-op — validate here).
_CACHE_CONFIG_KEYS = frozenset({
    "cache_type",
    "Fn_compute_blocks", "Bn_compute_blocks",
    "residual_diff_threshold", "max_accumulated_residual_diff_threshold",
    "max_warmup_steps", "warmup_interval",
    "max_cached_steps", "max_continuous_cached_steps",
    "enable_separate_cfg", "cfg_compute_first", "cfg_diff_compute_separate",
    "num_inference_steps",
    "steps_computation_mask", "steps_computation_policy",
    "force_refresh_step_hint", "force_refresh_step_policy",
    "enable_taylorseer", "taylorseer_order",
    "scm_steps_mask_policy", "scm_steps_policy",
})
DEFAULT_CACHE_CONFIG: dict = {   # values = the historical H3 deployment
    "Fn_compute_blocks": 1,
    "Bn_compute_blocks": 0,
    "max_warmup_steps": 4,
    "max_cached_steps": -1,
    "max_continuous_cached_steps": 1,
    "residual_diff_threshold": 0.04,   # official default; change only explicitly
    "enable_taylorseer": False,
    "taylorseer_order": 1,
    "scm_steps_mask_policy": None,
    "scm_steps_policy": "dynamic",
    "num_inference_steps": None,
    "force_refresh_step_hint": None,
    "force_refresh_step_policy": "once",
}


@dataclass
class DeployConfig:
    """Field names mirror the `vllm serve` CLI flags (dashes -> underscores),
    so the config reads exactly like the command line.

    Exceptions (not CLI flags):
      model                    positional arg of `vllm serve`
      cuda_visible_devices     the CUDA_VISIBLE_DEVICES env var;
                               --num-gpus is derived as len(devices)
      cache_config             the whole --cache-config JSON as a dict
                               (keys mirror vllm-omni's DiffusionCacheConfig)
      log_path / health_timeout_min  deployer-side, not passed to vllm
    """

    model: str = MODEL_DEFAULT
    # "auto" infers the checkpoint partition from the MODEL path
    # (FL2VA->fl2va, Ref2VA->ref2va, H3 root->combined) — mirrors the bash
    # deploy scripts. Without --task-type the server boots the fl2va
    # partition and rejects ref2va requests with HTTP 500.
    task_type: str = "auto"
    port: int = 9000
    cuda_visible_devices: str = "0,1,2,3"
    tensor_parallel_size: int = 4
    # None = auto: GPU count / tensor_parallel_size — the DiT parallel rule
    # every bash profile follows (TP x USP = GPUs; 4rtx5090: 4x1, 4h800:
    # 1x4, 8rtx5090: 4x2). Explicit values win; auto requires TP | GPUs.
    usp: int | None = None
    ring: int = 1
    # None = auto: full device width (len(cuda_visible_devices)). Every bash
    # deploy profile sets both to the GPU count — text encoder sharded across
    # all cards, VAE patch/tile-parallel across all cards — so the non-DiT
    # stages never bottleneck and leave DiT weight headroom per card.
    text_encoder_tp_size: int | None = None
    vae_patch_parallel_size: int | None = None
    num_weight_load_threads: int = 8
    diffusion_attention_backend: str = "CUDNN_ATTN"
    # "" (default) runs unquantized; "fp8" enables online fp8 (weight-only
    # via Marlin on pre-Hopper GPUs). Off by default: quantization is a
    # deliberate trade (memory/speed vs accuracy), not a silent default.
    quantization: str = ""
    # Off by default: CPU offload trades per-step PCIe traffic for GPU
    # memory headroom — enable deliberately on memory-tight setups.
    enable_cpu_offload: bool = False
    # Cache acceleration: "" (default) = no caching (clean baseline),
    # "cache_dit" enables it. Other backends (teacache, ...) are rejected
    # until cache_config's key validation covers their knobs. Caching trades
    # output fidelity for speed — opt in per profile like quantization.
    cache_backend: str = ""
    # Cache-DiT knobs as ONE dict (passed through to --cache-config). Keys
    # mirror vllm-omni's DiffusionCacheConfig so run snapshots stay directly
    # comparable with the official definitions. Constructed/swept as a dict:
    #   DeployConfig(cache_config={"enable_taylorseer": True, ...})
    # Partial dicts are fine — unknown keys fall back to engine defaults.
    cache_config: dict = field(default_factory=lambda: dict(DEFAULT_CACHE_CONFIG))
    log_path: str = "logs/deploy.log"
    pid_file: str = "logs/deploy.pid"   # written on --detach, read by --stop
    health_timeout_min: int = 15

    def __post_init__(self) -> None:
        num_gpus = len(self.cuda_visible_devices.split(","))
        if self.usp is None:
            if num_gpus % self.tensor_parallel_size:
                raise ValueError(
                    f"usp auto (= GPU count / tensor_parallel_size) needs "
                    f"tensor_parallel_size={self.tensor_parallel_size} to "
                    f"divide {num_gpus} GPUs; set usp explicitly")
            self.usp = num_gpus // self.tensor_parallel_size
        if self.text_encoder_tp_size is None:
            self.text_encoder_tp_size = num_gpus
        if self.vae_patch_parallel_size is None:
            self.vae_patch_parallel_size = num_gpus
        unknown = set(self.cache_config) - _CACHE_CONFIG_KEYS
        if self.cache_backend not in ("", "cache_dit"):
            raise ValueError(
                f"cache_backend must be '' (off) or 'cache_dit'; got "
                f"{self.cache_backend!r} (other backends land when "
                f"cache_config key validation covers their knobs)")
        if unknown:
            raise ValueError(
                f"cache_config has unknown key(s) {sorted(unknown)}; valid "
                f"keys are the official DiffusionCacheConfig names: "
                f"{sorted(_CACHE_CONFIG_KEYS)}")

    @classmethod
    def from_env(cls) -> "DeployConfig":
        def env(name: str, default: Any, cast: Callable[[str], Any] = str) -> Any:
            raw = os.environ.get(name, "")
            return cast(raw) if raw else default

        # CACHE_CONFIG: JSON string of cache-dit overrides merged over the
        # defaults, e.g. '{"enable_taylorseer": true}'.
        cache_config = dict(DEFAULT_CACHE_CONFIG)
        if os.environ.get("CACHE_CONFIG"):
            try:
                overrides = json.loads(os.environ["CACHE_CONFIG"])
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"CACHE_CONFIG is not valid JSON: {exc}\n"
                    f"  got: {os.environ['CACHE_CONFIG']!r}") from exc
            cache_config.update(overrides)

        return cls(
            model=env("MODEL", MODEL_DEFAULT),
            task_type=env("TASK_TYPE", "auto"),
            port=env("PORT", 9000, int),
            cuda_visible_devices=env("CUDA_VISIBLE_DEVICES", "0,1,2,3"),
            tensor_parallel_size=env("TENSOR_PARALLEL_SIZE", 4, int),
            usp=env("USP", None, int),
            ring=env("RING", 1, int),
            text_encoder_tp_size=env("TEXT_ENCODER_TP_SIZE", None, int),
            vae_patch_parallel_size=env("VAE_PATCH_PARALLEL_SIZE", None, int),
            num_weight_load_threads=env("NUM_WEIGHT_LOAD_THREADS", 8, int),
            diffusion_attention_backend=env("DIFFUSION_ATTENTION_BACKEND", "CUDNN_ATTN"),
            quantization=env("QUANTIZATION", ""),
            enable_cpu_offload=env("ENABLE_CPU_OFFLOAD", "0") == "1",
            cache_backend=env("CACHE_BACKEND", ""),
            cache_config=cache_config,
            log_path=env("LOG", "logs/deploy.log"),
            pid_file=env("PID_FILE", "logs/deploy.pid"),
            health_timeout_min=env("HEALTH_TIMEOUT_MIN", 15, int),
        )

    @classmethod
    def from_config(cls, overrides: dict) -> "DeployConfig":
        """Config-file constructor: from_env() plus a dict of key overrides.

        Symmetric to from_env() (env still fills anything the dict omits;
        the dict wins per key). Unknown keys are rejected so a typo fails
        loudly instead of silently no-op'ing. A partial cache_config dict
        MERGES over the resolved value (same semantics as CACHE_CONFIG).
        """
        cfg = cls.from_env()
        if not isinstance(overrides, dict):
            raise TypeError(f"deploy overrides must be a dict, got "
                            f"{type(overrides).__name__}")
        unknown = set(overrides) - {f.name for f in dataclasses.fields(cfg)}
        if unknown:
            raise ValueError(f"deploy config has unknown key(s) "
                             f"{sorted(unknown)}; valid keys are the "
                             f"DeployConfig field names: "
                             f"{sorted(f.name for f in dataclasses.fields(cfg))}")
        kw = {k: v for k, v in overrides.items() if k != "cache_config"}
        if "cache_config" in overrides:
            merged = dict(cfg.cache_config)
            merged.update(overrides["cache_config"])
            kw["cache_config"] = merged
        # replace() re-runs __post_init__, re-validating cache_config keys.
        return dataclasses.replace(cfg, **kw)

    def build_cmd(self) -> list[str]:
        """config -> vllm serve CLI (single source of truth for the mapping)."""
        cache_config = json.dumps(self.cache_config)
        num_gpus = len(self.cuda_visible_devices.split(","))
        cmd = [
            "vllm", "serve", self.model,
            "--omni",
            "--task-type", self.task_type,
            "--trust-remote-code",
            "--host", "0.0.0.0", 
            "--port", str(self.port),
            "--num-gpus", str(num_gpus),
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--usp", str(self.usp), 
            "--ring", str(self.ring),
            "--text-encoder-tp-size", str(self.text_encoder_tp_size),
            "--vae-patch-parallel-size", str(self.vae_patch_parallel_size),
            "--vae-parallel-mode", "tile", 
            "--vae-use-tiling",
            "--num-weight-load-threads", str(self.num_weight_load_threads),
            "--diffusion-compile-granularity", "regional",
            "--diffusion-attention-backend", self.diffusion_attention_backend,
        ]
        # Cache acceleration is optional: "" disables it entirely (baseline
        # runs); the only accepted backend today is cache_dit (validated in
        # __post_init__), which also gets the per-step summary flag.
        if self.cache_backend:
            cmd += ["--cache-backend", self.cache_backend,
                    "--cache-config", cache_config,
                    "--enable-cache-dit-summary"]
        if self.quantization:
            cmd += ["--quantization", self.quantization]
        if self.enable_cpu_offload:
            cmd += ["--enable-cpu-offload"]
        return cmd

    def build_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        env["VLLM_OMNI_VIDEO_SYNC_TIMEOUT"] = "4500"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        return env


def _workers_left() -> list[str]:
    try:
        out = subprocess.run(
            ["pgrep", "-af", "DiffusionWorker"],
            capture_output=True, 
            text=True, 
            timeout=10,
        ).stdout.strip()
        return [line for line in out.splitlines() if line]
    except Exception:
        return []


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)  # signal 0 = existence probe only
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by someone else


def graceful_stop_pid(pid: int, wait_sec: int = 120) -> bool:
    """TERM -> wait -> KILL as last resort -> verify no worker remnants.

    Works for any pid (not just a Deployer-owned child), so a --stop
    invocation can stop a service started by an earlier --detach.

    Hard-killing a rank mid-collective orphans NCCL spin kernels that peg the
    GPUs at 100% until a driver reset; never skip the graceful path.
    """
    if _pid_alive(pid):
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + wait_sec
        while _pid_alive(pid) and time.monotonic() < deadline:
            time.sleep(1)
        if _pid_alive(pid):
            print(f"[deploy] WARNING: pid {pid} still alive after "
                  f"{wait_sec}s, sending KILL", file=sys.stderr)
            os.kill(pid, signal.SIGKILL)
            deadline = time.monotonic() + 30
            while _pid_alive(pid) and time.monotonic() < deadline:
                time.sleep(1)
    leftovers = _workers_left()
    if leftovers:
        print(f"[deploy] WARNING: orphaned workers: {leftovers}", file=sys.stderr)
        return False
    print("[deploy] stopped cleanly, no worker remnants")
    return True


class Deployer:
    """Owns one service lifecycle: serve -> healthy -> (traffic) -> stop.

    Usable as a context manager; pairs with generate.py's Generator via
    `Generator.from_deployer(deployer)`, which reads `deployer.port` to
    target the traffic.

        with Deployer(cfg) as d:          # serve() + wait_healthy()
            gen = Generator.from_deployer(d)
            gen.run()
        # __exit__ stops the service gracefully even on exceptions
    """

    def __init__(self, cfg: DeployConfig):
        self.cfg = cfg
        self.proc: subprocess.Popen | None = None

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> "Deployer":
        """Launch the service and wait until /health is ready."""
        self.serve()
        if not self.wait_healthy():
            self.stop()
            raise RuntimeError(
                f"service on port {self.cfg.port} failed to become healthy "
                f"(see {self.cfg.log_path})")
        return self

    def serve(self) -> subprocess.Popen:
        """Launch the background service (no health wait).

        start_new_session=True detaches the child from this terminal's
        SIGHUP (SSH drop won't kill the service); stopping it must go
        through stop()/terminate().
        """
        if self.proc is not None and self.proc.poll() is None:
            raise RuntimeError(f"already serving pid={self.proc.pid}")
        os.makedirs(os.path.dirname(self.cfg.log_path) or ".", exist_ok=True)
        cmd = self.cfg.build_cmd()
        print(f"[deploy] {' '.join(shlex.quote(c) for c in cmd)}")
        print(f"[deploy] log: {self.cfg.log_path}")
        log_fh = open(self.cfg.log_path, "w", buffering=1)
        self.proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=self.cfg.build_env(),
            start_new_session=True,
        )
        return self.proc

    def wait_healthy(self, timeout_min: int | None = None) -> bool:
        """Poll /health until ready. Returns False (and prints FATAL) if the
        launcher dies or the timeout elapses."""
        if self.proc is None:
            raise RuntimeError("not serving; call serve()/start() first")
        port = self.cfg.port
        timeout_min = timeout_min or self.cfg.health_timeout_min
        deadline = time.monotonic() + timeout_min * 60
        url = f"http://localhost:{port}/health"
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                print(f"[deploy] FATAL: launcher exited rc={self.proc.returncode}",
                      file=sys.stderr)
                return False
            try:
                with urllib.request.urlopen(url, timeout=3) as resp:
                    if resp.status == 200:
                        print(f"[deploy] healthy: {url}")
                        return True
            except Exception:
                pass
            time.sleep(5)
        print(f"[deploy] FATAL: health check timed out after {timeout_min}min",
              file=sys.stderr)
        return False

    def stop(self, wait_sec: int = 120) -> bool:
        """Graceful stop (TERM -> wait -> KILL -> verify no worker remnants)."""
        if self.proc is None:
            return True  # nothing we started; nothing to stop
        ok = graceful_stop_pid(self.proc.pid, wait_sec=wait_sec)
        self.proc = None
        return ok

    # -- service facts (for the generation side) ------------------------------

    @property
    def port(self) -> int:
        return self.cfg.port

    @property
    def pid(self) -> int | None:
        return self.proc.pid if self.proc is not None else None

    @property
    def log_path(self) -> str:
        return self.cfg.log_path

    # -- context manager -------------------------------------------------------

    def __enter__(self) -> "Deployer":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.stop()
        return False  # never swallow exceptions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--detach", action="store_true",
                    help="background + wait healthy, then exit (service keeps running; "
                         "pid recorded in the pid file for --stop)")
    parser.add_argument("--stop", action="store_true",
                    help="gracefully stop the service started by --detach "
                         "(reads the pid from the pid file, then TERM -> wait -> KILL "
                         "-> verify no worker remnants)")
    parser.add_argument("--stop-pid", type=int, default=None, metavar="PID",
                    help="stop this pid instead of the one in the pid file")
    parser.add_argument("--dry-run", action="store_true",
                    help="print the vllm command and exit")
    args = parser.parse_args()

    cfg = DeployConfig.from_env()

    if args.stop or args.stop_pid is not None:
        pid = args.stop_pid
        if pid is None:
            try:
                with open(cfg.pid_file) as fh:
                    pid = int(fh.read().strip())
            except (OSError, ValueError):
                print(f"[deploy] FATAL: cannot read pid from {cfg.pid_file}; "
                      f"use --stop-pid <PID> explicitly", file=sys.stderr)
                return 1
        print(f"[deploy] stopping pid {pid} ...")
        return 0 if graceful_stop_pid(pid) else 1
    if args.dry_run:
        print(" ".join(shlex.quote(c) for c in cfg.build_cmd()))
        return 0

    deployer = Deployer(cfg)
    proc = deployer.serve()
    if args.detach:
        if not deployer.wait_healthy():
            deployer.stop()
            return 1
        with open(cfg.pid_file, "w") as fh:
            fh.write(f"{deployer.pid}\n")
        print(f"[deploy] detached; pid={deployer.pid} log={deployer.log_path} "
              f"port={cfg.port} pid_file={cfg.pid_file}")
        return 0

    # Foreground mode: forward Ctrl-C/SIGTERM to the service, wait for exit.
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: proc.terminate())
    try:
        rc = proc.wait()
        print(f"[deploy] exited rc={rc}")
        return rc
    except KeyboardInterrupt:
        print("\n[deploy] interrupt -> graceful stop")
        deployer.stop()
        return 130


if __name__ == "__main__":
    sys.exit(main())
