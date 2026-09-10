#!/usr/bin/env python3
"""Fan-out the FL2VA request to N services concurrently, R rounds.

Python port of scripts/MiniMax-H3/generate/generate.sh (drop-in replacement:
same env knobs, same request shape, same output naming).

Two ways to use (same layout as deploy.py):

1. As a module (for the grid-search driver):
     from generate import GenerateConfig, Generator
     ok = Generator(GenerateConfig.from_env()).run()

   or paired with a service owned by deploy.py's Deployer:
     with Deployer(DeployConfig(...)) as d:       # serve + wait healthy
         ok = Generator.from_deployer(d, duration=15).run()
         ...                                     # client-side: gen.results

2. As a CLI:
     python generate.py                   # INPUT_DIR + env knobs, see below

Env knobs (defaults match the bash version unless noted):
  PORT_BASE      base port for contiguous derivation        (9000)
  NUM_SERVICES   number of services                          (1)
  PORTS          explicit space-separated port list; overrides base/count
  HOST           service host                                 (localhost)
  OUT_DIR        output directory                             (./outputs)
  ROUNDS         rounds of concurrent fan-out                 (5)
  SEED           generation seed                              (0)
  TASK_TYPE      extra_params task: fl2va | ref2va           (fl2va)
  DURATION       audio/video seconds in extra_params          (5)
  WIDTH          explicit output width                        (832)
  HEIGHT         explicit output height                       (480)
  INPUT_DIR      the ONLY input knob: per-case directory holding prompt.txt
                 plus reference files (sorted filename order = upload order,
                 which defines the <Picture/Video N> numbering in the prompt).
                 fl2va: 0-2 reference frame images, default <repo>/inputs/i2va.
                 ref2va: mixed images/videos/audios (<=9 img, <=3 vid, <=3 aud,
                 <=12 total), default <repo>/inputs/r2va.
  USE_IR_PROMPT  generate from prompt_ir.txt (the H3-Context-IR enhanced
                 prompt, see minimax_h3.py) instead of prompt.txt (false)
  REQUEST_TIMEOUT  per-request read timeout, seconds         (1800)

Notes:
- prompt_ir.txt (minimax_h3.py's output artifact) is never uploaded as a
  reference; USE_IR_PROMPT only switches which file the prompt comes from.
- The request is multipart form -> POST http://HOST:PORT/v1/videos/sync,
  exactly mirroring the curl -F fields of the bash clients: fl2va repeats
  "input_reference" image fields; ref2va repeats "input_references" fields
  whose modality the server detects from each file's MIME type.
- Per-request wall time is measured client-side (in addition to the
  server-side e2e_total_ms you can grep from service logs).
"""

from __future__ import annotations

import json
import dataclasses
import mimetypes
import os
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


@dataclass
class GenerateConfig:
    """Fan-out request shape + inputs (mirrors DeployConfig's layout).

    Field names follow the env knobs / bash client; exceptions (not request
    fields):
      prompt / prompt_file / ref_files   resolved from INPUT_DIR
      ports                              derived from PORT_BASE/NUM_SERVICES
                                         unless PORTS is explicit
      request_timeout                    client-side knob, not sent
    """

    host: str = "localhost"
    port_base: int = 9000
    num_services: int = 1
    out_dir: str = "./outputs"
    rounds: int = 5
    seed: str = "0"
    task_type: str = "fl2va"
    duration: int = 5
    width: int = 832
    height: int = 480
    input_dir: str = os.path.join(REPO_ROOT, "inputs", "i2va")
    use_ir_prompt: bool = False
    request_timeout: float = 4500.0

    # Derived in __post_init__.
    ports: list[int] = field(default_factory=list)
    prompt_file: str = ""
    prompt: str = ""
    ref_files: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "GenerateConfig":
        def env(name: str, default: Any, cast: Callable[[str], Any] = str) -> Any:
            raw = os.environ.get(name, "")
            return cast(raw) if raw else default

        ports_env = os.environ.get("PORTS", "")
        # Only INPUT_DIR is task-aware (fl2va -> inputs/i2va, ref2va ->
        # inputs/r2va); all other defaults are shared between tasks.
        task_type = env("TASK_TYPE", "fl2va")
        return cls(
            host=env("HOST", "localhost"),
            port_base=env("PORT_BASE", 9000, int),
            num_services=env("NUM_SERVICES", 1, int),
            ports=[int(p) for p in ports_env.split()] if ports_env else [],
            out_dir=env("OUT_DIR", "./outputs"),
            rounds=env("ROUNDS", 5, int),
            seed=env("SEED", "0"),
            task_type=task_type,
            duration=env("DURATION", 5, int),
            width=env("WIDTH", 832, int),
            height=env("HEIGHT", 480, int),
            input_dir=env("INPUT_DIR", os.path.join(
                REPO_ROOT, "inputs", "r2va" if task_type == "ref2va"
                else "i2va")),
            use_ir_prompt=env("USE_IR_PROMPT", False,
                              lambda v: v.strip().lower()
                              in ("1", "true", "yes", "on")),
            request_timeout=env("REQUEST_TIMEOUT", 4500.0, float),
        )

    # Fields __post_init__ re-derives from input_dir — they are snapshot
    # OUTPUT, not config input. Overriding them would be silently discarded.
    DERIVED_FIELDS = frozenset({"prompt", "prompt_file", "ref_files"})

    @classmethod
    def from_config(cls, overrides: dict) -> "GenerateConfig":
        """Config-file constructor: from_env() plus a dict of key overrides.

        Symmetric to from_env() (env still fills anything the dict omits;
        the dict wins per key). Unknown keys are rejected so a typo fails
        loudly. replace() re-runs __post_init__, so derived fields
        (prompt/ref_files) reload against the overridden input_dir/task_type;
        passing those derived keys explicitly is not an error (run snapshots
        contain them) but they are stripped with a warning — edit input_dir
        instead.
        """
        cfg = cls.from_env()
        if not isinstance(overrides, dict):
            raise TypeError(f"generate overrides must be a dict, got "
                            f"{type(overrides).__name__}")
        derived = set(overrides) & cls.DERIVED_FIELDS
        if derived:
            print(f"[generate] WARNING: ignoring derived key(s) {sorted(derived)} "
                  f"from config — they are re-derived from input_dir; "
                  f"set input_dir instead", file=sys.stderr)
            overrides = {k: v for k, v in overrides.items()
                         if k not in cls.DERIVED_FIELDS}
        unknown = set(overrides) - {f.name for f in dataclasses.fields(cfg)}
        if unknown:
            raise ValueError(f"generate config has unknown key(s) "
                             f"{sorted(unknown)}; valid keys are the "
                             f"GenerateConfig field names: "
                             f"{sorted(f.name for f in dataclasses.fields(cfg))}")
        return dataclasses.replace(cfg, **overrides)

    def __post_init__(self) -> None:
        if not self.ports:
            self.ports = [self.port_base + i for i in range(self.num_services)]
        # The ONLY input knob is INPUT_DIR: prompt.txt plus 0-2 reference
        # frame images (sorted filename order = upload order). With
        # use_ir_prompt, the prompt comes from the H3-Context-IR enhanced
        # prompt_ir.txt instead (produced by minimax_h3.py on this dir).
        self.prompt_file = os.path.join(
            self.input_dir,
            "prompt_ir.txt" if self.use_ir_prompt else "prompt.txt")
        if not os.path.isfile(self.prompt_file):
            hint = (" — run minimax_h3.py on this INPUT_DIR first to "
                    "produce the enhanced prompt"
                    if self.use_ir_prompt else " (check INPUT_DIR)")
            raise FileNotFoundError(f"{self.prompt_file} not found{hint}")
        with open(self.prompt_file, encoding="utf-8") as fh:
            self.prompt = fh.read()
        try:
            names = sorted(os.listdir(self.input_dir))
        except OSError as exc:
            raise OSError(f"cannot read INPUT_DIR {self.input_dir}: {exc}") from exc
        # prompt_ir.txt (IR-enhanced prompt) and *.json (h3_context_ir.py
        # trace files) are never references; .json is not an accepted
        # modality anyway.
        self.ref_files = [os.path.join(self.input_dir, n) for n in names
                          if n not in ("prompt.txt", "prompt_ir.txt")
                          and not n.startswith("README")
                          and os.path.splitext(n)[1].lower() != ".json"]
        self._validate_refs()

    # Ref2VA contract (enforced server-side, checked here for a clearer
    # client-side error): <=9 images, <=3 videos, <=3 audios, <=12 total.
    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm"}
    AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a"}

    def _ref_kind(self, path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        if ext in self.IMAGE_EXTS:
            return "image"
        if ext in self.VIDEO_EXTS:
            return "video"
        if ext in self.AUDIO_EXTS:
            return "audio"
        raise ValueError(f"unrecognized reference modality for '{path}' "
                         f"(extension '{ext}' is not image/video/audio)")

    def _validate_refs(self) -> None:
        if self.task_type == "ref2va":
            kinds = [self._ref_kind(p) for p in self.ref_files]
            counts = {k: kinds.count(k) for k in ("image", "video", "audio")}
            limits = {"image": 9, "video": 3, "audio": 3}
            for kind, limit in limits.items():
                if counts[kind] > limit:
                    raise ValueError(f"ref2va allows at most {limit} {kind} "
                                     f"reference(s), got {counts[kind]}")
            if len(self.ref_files) > 12:
                raise ValueError(f"ref2va allows at most 12 reference files, "
                                 f"got {len(self.ref_files)}")
        else:
            for p in self.ref_files:
                kind = self._ref_kind(p)
                if kind != "image":
                    raise ValueError(f"{self.task_type} reference files must "
                                     f"be images, got {kind}: {p}")
            if len(self.ref_files) > 2:
                raise ValueError(f"INPUT_DIR holds more than 2 reference frame "
                                 f"images (first [+ last]): {self.ref_files}")

    @property
    def ref_desc(self) -> str:
        if self.task_type == "ref2va":
            n = len(self.ref_files)
            return f"{n} reference file(s) ({'/'.join(sorted(set(
                self._ref_kind(p) for p in self.ref_files))) or 'none'})"
        return {
            0: "0 reference frames (text-only)",
            1: "first frame only",
            2: "first + last frame",
        }[len(self.ref_files)]

    def build_form(self) -> dict[str, str]:
        """config -> multipart form fields (single source of truth, mirrors
        the curl -F flags of the bash client)."""
        return {
            "prompt": self.prompt,
            "fps": "24",
            "num_inference_steps": "50",
            "flow_shift": "12",
            "seed": self.seed,
            "width": str(self.width),
            "height": str(self.height),
            "extra_params": json.dumps({
                "task": self.task_type,
                "duration": int(self.duration),
                "audio_flow_shift": 3.0,
            }),
        }

    def out_path(self, rnd: int, svc: int, port: int) -> str:
        return os.path.join(
            self.out_dir,
            f"{self.task_type}_r{rnd}_svc{svc}_port{port}_seed{self.seed}.mp4",
        )


# ---------------------------------------------------------------------------
# generation behavior
# ---------------------------------------------------------------------------

class Generator:
    """Fan-out generation driver: one instance drives ROUNDS x PORTS requests
    against a running service (or services).

    Pairs with deploy.py's Deployer: construct via `Generator.from_deployer(d)`
    to target the single service a Deployer started, e.g.

        with Deployer(DeployConfig(...)) as d:      # serve + wait healthy
            gen = Generator.from_deployer(d, duration=15)
            gen.run()                               # 5-round fan-out
            ...                                     # d stops on exit

    `results` accumulates one record per request ({round, svc, port, ok,
    code, elapsed, error}) so callers (e.g. the grid-search driver) can read
    client-side timings after run().
    """

    def __init__(self, cfg: GenerateConfig):
        self.cfg = cfg
        self.results: list[dict] = []

    @classmethod
    def from_deployer(cls, deployer, **overrides) -> "Generator":
        """Target the service started by a deploy.py Deployer.

        Takes anything with a `.port` attribute (Deployer or DeployConfig),
        keeping the dependency one-directional: generate.py knows nothing
        about deploy.py's types.
        """
        cfg = GenerateConfig(ports=[deployer.port], **overrides)
        return cls(cfg)

    def _post_one(self, port: int, out_path: str) -> dict:
        """Send one request; returns a result record."""
        url = f"http://{self.cfg.host}:{port}/v1/videos/sync"
        form = self.cfg.build_form()
        # Repeated file field per reference (upload order = <Picture/Video N>
        # numbering in the prompt). Field name and MIME follow the task:
        # fl2va/t2va send "input_reference" image frames; ref2va sends
        # "input_references" mixed image/video/audio files whose modality the
        # server detects from the MIME type.
        field = "input_references" if self.cfg.task_type == "ref2va" \
            else "input_reference"
        frame_handles = [open(p, "rb") for p in self.cfg.ref_files]
        files = [(field, (os.path.basename(p), fh,
                          mimetypes.guess_type(p)[0] or "image/png"))
                 for p, fh in zip(self.cfg.ref_files, frame_handles)]
        started = time.monotonic()
        try:
            resp = requests.post(url, data=form, files=files,
                                 timeout=self.cfg.request_timeout)
            elapsed = time.monotonic() - started
            body = resp.content
            with open(out_path, "wb") as fh:
                fh.write(body)
            if resp.status_code == 200 and body:
                return {"ok": True, "code": resp.status_code,
                        "elapsed": elapsed, "error": ""}
            return {"ok": False, "code": resp.status_code,
                    "elapsed": elapsed,
                    "error": f"non-200 or empty body (see {out_path})"}
        except requests.RequestException as exc:
            elapsed = time.monotonic() - started
            reason = str(exc) or exc.__class__.__name__
            return {"ok": False, "code": 0, "elapsed": elapsed, "error": reason}
        finally:
            for fh in frame_handles:
                fh.close()

    @staticmethod
    def _ffprobe_summary(out_path: str) -> str:
        if shutil.which("ffprobe") is None:
            return ""
        try:
            proc = subprocess.run(
                ["ffprobe",
                 "-v", "error",
                 "-select_streams", "v:0",
                 "-show_entries", "stream=width,height,nb_frames,r_frame_rate",
                 "-of", "csv=p=0",
                 out_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return (proc.stdout.strip().splitlines()[0]
                    if proc.stdout.strip() else "")
        except Exception:
            return ""

    def run(self) -> bool:
        """Drive the ROUNDS x PORTS fan-out; returns True iff all requests OK."""
        cfg = self.cfg
        os.makedirs(cfg.out_dir, exist_ok=True)
        ports_str = " ".join(str(p) for p in cfg.ports)
        print(f"[generate.py] prompt:  {cfg.prompt_file}")
        print(f"[generate.py] frames:  "
              f"{' '.join(cfg.ref_files) or '<none — text-only request>'}\n")
        print(f"Posting {cfg.width}x{cfg.height}/{cfg.duration}s "
              f"{cfg.task_type} request ({cfg.ref_desc}) to "
              f"{len(cfg.ports)} service(s): {ports_str}, {cfg.rounds} round(s) "
              f"(concurrent fan-out per round)...\n")

        fail = False
        pool = ThreadPoolExecutor(max_workers=len(cfg.ports))
        try:
            for rnd in range(1, cfg.rounds + 1):
                print(f"=== Round {rnd}/{cfg.rounds} ===")
                jobs = []
                for i, port in enumerate(cfg.ports):
                    out = cfg.out_path(rnd, i, port)
                    print(f"  service {i} -> http://{cfg.host}:{port}  -> {out}")
                    jobs.append((i, port, out,
                                 pool.submit(self._post_one, port, out)))

                print(f"\nWaiting for all {len(cfg.ports)} request(s) of round "
                      f"{rnd} to finish...")
                print(f"Round {rnd} results:")
                for i, port, out, fut in jobs:
                    res = fut.result()
                    self.results.append({"round": rnd, "svc": i, "port": port,
                                         "out": out, **res})
                    if res["ok"]:
                        line = (f"  [OK]  r{rnd} svc{i} port={port} "
                                f"-> {out}  [{res['elapsed']:.1f}s client]")
                        meta = self._ffprobe_summary(out)
                        if meta:
                            line += f"  [{meta}]"
                        print(line)
                    else:
                        print(f"  [FAIL] r{rnd} svc{i} port={port} "
                              f"http={res['code']} elapsed={res['elapsed']:.1f}s "
                              f"error={res['error']}")
                        fail = True
                print()
        except (KeyboardInterrupt, SystemExit):
            # A Ctrl+C must not hang the run: the executor's exit hook joins
            # worker threads at interpreter shutdown, and one in-flight
            # request can hold the process for its full read timeout
            # (REQUEST_TIMEOUT, up to 4500s). Cancel what hasn't started,
            # skip the join, and re-raise so the pipeline stops the service
            # and hard-exits. SIGINT is shielded first: a second Ctrl+C
            # during the service's TERM->KILL cleanup would abort it.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            pool.shutdown(wait=False, cancel_futures=True)
            print("\n[generate.py] interrupted — abandoning in-flight "
                  "request(s); the pipeline will stop the service",
                  file=sys.stderr)
            raise
        pool.shutdown(wait=True)

        if fail:
            print("Some requests FAILED.")
            return False
        print(f"All done: {cfg.rounds} rounds x {len(cfg.ports)} service(s). "
              f"Outputs in {cfg.out_dir}/ "
              f"({cfg.task_type}_r<R>_svc<N>_...)")
        print("Read steady-state e2e_total_ms from each service log from "
              "round ~3 onward (rounds 1-2 are warmup/settling).")
        return True


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    try:
        cfg = GenerateConfig.from_env()
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0 if Generator(cfg).run() else 1


if __name__ == "__main__":
    sys.exit(main())
