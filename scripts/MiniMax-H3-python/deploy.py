#!/usr/bin/env python3
"""MiniMax-H3 FL2VA service deployer (Python port of the rtx5090 deploy.sh).

Two ways to use:

1. As a module (for pipeline.py / search.py):
     from deploy import DeployConfig, serve, wait_healthy, graceful_stop
     cfg = DeployConfig.from_env()          # or construct explicitly / override
     proc, log_path = serve(cfg)            # background, log to file
     ok = wait_healthy(proc, cfg.port)      # poll /health, abort if launcher dies
     ...
     graceful_stop(proc)                    # TERM -> wait -> KILL -> verify clean

2. As a CLI (parity with deploy.sh):
     python deploy.py                 # foreground, Ctrl-C stops the service
     python deploy.py --detach        # background + log + wait healthy, then exit
                                      # (service keeps running; stop via --stop-pid file)

Config knobs (env names = field names uppercased; defaults mirror 4rtx5090/deploy.sh):
  MODEL                      model path
  PORT                       service port                    (9000)
  CUDA_VISIBLE_DEVICES       gpu list                        (0,1,2,3)
  TENSOR_PARALLEL_SIZE       (4)
  TEXT_ENCODER_TP_SIZE       (4)
  USP / RING                 (1 / 1)
  VAE_PATCH_PARALLEL_SIZE    (4)
  QUANTIZATION               "fp8" or "" to disable          (fp8)
  ENABLE_CPU_OFFLOAD         1/0                             (1)
  DIFFUSION_ATTENTION_BACKEND                                 (CUDNN_ATTN)
  RESIDUAL_DIFF_THRESHOLD    Cache-DiT threshold             (0.04 — official
                             default; per project policy only change explicitly)
  NUM_WEIGHT_LOAD_THREADS    (8)
  LOG / HEALTH_TIMEOUT_MIN   deployer-side knobs

Every deploy also exports: VLLM_WORKER_MULTIPROC_METHOD=spawn,
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
(anti-fragmentation; 768p OOMed without it).
"""

from __future__ import annotations

import argparse
import dataclasses
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


@dataclass
class DeployConfig:
    """Field names mirror the `vllm serve` CLI flags (dashes -> underscores),
    so the config reads exactly like the command line.

    Exceptions (not CLI flags):
      model                    positional arg of `vllm serve`
      cuda_visible_devices     the CUDA_VISIBLE_DEVICES env var;
                               --num-gpus is derived as len(devices)
      residual_diff_threshold  lives inside --cache-config JSON
      log_path / health_timeout_min  deployer-side, not passed to vllm
    """

    model: str = MODEL_DEFAULT
    port: int = 9000
    cuda_visible_devices: str = "0,1,2,3"
    tensor_parallel_size: int = 4
    usp: int = 1
    ring: int = 1
    text_encoder_tp_size: int = 4
    vae_patch_parallel_size: int = 4
    num_weight_load_threads: int = 8
    diffusion_attention_backend: str = "CUDNN_ATTN"
    quantization: str = "fp8"           # "" disables the flag
    enable_cpu_offload: bool = True
    residual_diff_threshold: float = 0.04   # official default; change only explicitly
    log_path: str = "logs/deploy_py.log"
    health_timeout_min: int = 15

    @classmethod
    def from_env(cls) -> "DeployConfig":
        def env(name: str, default: Any, cast: Callable[[str], Any] = str) -> Any:
            raw = os.environ.get(name, "")
            return cast(raw) if raw else default

        return cls(
            model=env("MODEL", MODEL_DEFAULT),
            port=env("PORT", 9000, int),
            cuda_visible_devices=env("CUDA_VISIBLE_DEVICES", "0,1,2,3"),
            tensor_parallel_size=env("TENSOR_PARALLEL_SIZE", 4, int),
            usp=env("USP", 1, int),
            ring=env("RING", 1, int),
            text_encoder_tp_size=env("TEXT_ENCODER_TP_SIZE", 4, int),
            vae_patch_parallel_size=env("VAE_PATCH_PARALLEL_SIZE", 4, int),
            num_weight_load_threads=env("NUM_WEIGHT_LOAD_THREADS", 8, int),
            diffusion_attention_backend=env("DIFFUSION_ATTENTION_BACKEND", "CUDNN_ATTN"),
            quantization=env("QUANTIZATION", "fp8"),
            enable_cpu_offload=env("ENABLE_CPU_OFFLOAD", "1") == "1",
            residual_diff_threshold=env("RESIDUAL_DIFF_THRESHOLD", 0.04, float),
            log_path=env("LOG", "logs/deploy_py.log"),
            health_timeout_min=env("HEALTH_TIMEOUT_MIN", 15, int),
        )

    def build_cmd(self) -> list[str]:
        """config -> vllm serve CLI (single source of truth for the mapping)."""
        cache_config = (
            '{"Fn_compute_blocks":1,"Bn_compute_blocks":0,"max_warmup_steps":4,'
            f'"residual_diff_threshold":{self.residual_diff_threshold},'
            '"max_continuous_cached_steps":1,"enable_taylorseer":false}'
        )
        num_gpus = len(self.cuda_visible_devices.split(","))
        cmd = [
            "vllm", "serve", self.model,
            "--omni", 
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
            "--cache-backend", "cache_dit",
            "--cache-config", cache_config,
            "--enable-cache-dit-summary",
        ]
        if self.quantization:
            cmd += ["--quantization", self.quantization]
        if self.enable_cpu_offload:
            cmd += ["--enable-cpu-offload"]
        return cmd

    def build_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        env["VLLM_OMNI_VIDEO_SYNC_TIMEOUT"] = "1800"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        return env


def serve(cfg: DeployConfig):
    """Start vllm serve in the background; returns (Popen, log_path).

    start_new_session=True detaches the child from this terminal's SIGHUP
    (SSH drop won't kill the service); stopping it must go through
    graceful_stop()/terminate().
    """
    os.makedirs(os.path.dirname(cfg.log_path) or ".", exist_ok=True)
    cmd = cfg.build_cmd()
    print(f"[deploy] {' '.join(shlex.quote(c) for c in cmd)}")
    print(f"[deploy] log: {cfg.log_path}")
    log_fh = open(cfg.log_path, "w", buffering=1)
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        env=cfg.build_env(),
        start_new_session=True,
    )
    return proc, cfg.log_path


def wait_healthy(proc: subprocess.Popen, port: int, timeout_min: int = 15) -> bool:
    """Poll /health until ready. Returns False (and prints log tail) if the
    launcher dies or the timeout elapses."""
    deadline = time.monotonic() + timeout_min * 60
    url = f"http://localhost:{port}/health"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            print(f"[deploy] FATAL: launcher exited rc={proc.returncode}", file=sys.stderr)
            return False
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    print(f"[deploy] healthy: {url}")
                    return True
        except Exception:
            pass
        time.sleep(5)
    print(f"[deploy] FATAL: health check timed out after {timeout_min}min", file=sys.stderr)
    return False


def _workers_left() -> list[str]:
    try:
        out = subprocess.run(
            ["pgrep", "-af", "DiffusionWorker"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return [line for line in out.splitlines() if line]
    except Exception:
        return []


def graceful_stop(proc: subprocess.Popen, wait_sec: int = 120) -> bool:
    """TERM -> wait -> KILL as last resort -> verify no worker remnants.

    Hard-killing a rank mid-collective orphans NCCL spin kernels that peg the
    GPUs at 100% until a driver reset; never skip the graceful path.
    """
    if proc.poll() is not None:
        pass  # already exited; still check for orphaned workers below
    else:
        proc.terminate()
        try:
            proc.wait(timeout=wait_sec)
        except subprocess.TimeoutExpired:
            print("[deploy] WARNING: still alive after "
                  f"{wait_sec}s, sending KILL", file=sys.stderr)
            proc.kill()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
    leftovers = _workers_left()
    if leftovers:
        print(f"[deploy] WARNING: orphaned workers: {leftovers}", file=sys.stderr)
        return False
    print("[deploy] stopped cleanly, no worker remnants")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--detach", action="store_true",
                    help="background + wait healthy, then exit (service keeps running)")
    parser.add_argument("--dry-run", action="store_true",
                    help="print the vllm command and exit")
    args = parser.parse_args()

    cfg = DeployConfig.from_env()
    if args.dry_run:
        print(" ".join(shlex.quote(c) for c in cfg.build_cmd()))
        return 0

    proc, log_path = serve(cfg)
    if args.detach:
        if not wait_healthy(proc, cfg.port, cfg.health_timeout_min):
            graceful_stop(proc)
            return 1
        print(f"[deploy] detached; pid={proc.pid} log={log_path} port={cfg.port}")
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
        graceful_stop(proc)
        return 130


if __name__ == "__main__":
    sys.exit(main())
