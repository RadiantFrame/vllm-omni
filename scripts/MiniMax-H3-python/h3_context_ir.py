#!/usr/bin/env python3
"""One-shot MiniMax H3-Context-IR client: submit -> poll -> print result.

Wraps the two-step MiniMax API (create a H3-Context-IR task, then poll the
video-generation query endpoint) into a single blocking call: the script
exits when the task reaches a terminal state and prints the final result
JSON. Unlike video generation, the result is text (modality: "text"): the
enhanced prompt / context IR lives in task.content.prompt, ready to feed
the follow-up video-generation call.

Usage:
    MINIMAX_API_KEY=... python h3_context_ir.py [--task-type TYPE]
                                               [--input-dir DIR]
                                               [--ir-file FILE]
                                               [--duration S] [--ratio W:H]

Three request modes (TASK_TYPE / --task-type), mirroring the API's content
combinations. Inputs come from INPUT_DIR (same convention as generate.py):
prompt.txt plus reference files, sorted filename order = upload order (the
<Picture/Video N> numbering the prompt refers to); local files are sent as
base64 data URLs.

  t2va  text-only             no reference files; RATIO required and
                              non-adaptive (default 16:9); default
                              INPUT_DIR <repo>/inputs/t2va
  i2va  first [+last] frame   1-2 images; ratio is always adaptive (the
                              frame decides it); default INPUT_DIR
                              <repo>/inputs/i2va
  r2va  multimodal reference  >=1 and <=9 images + <=3 videos + <=3 audios,
                              role reference_<kind>; ratio optional (default
                              adaptive); default INPUT_DIR <repo>/inputs/r2va

On success the enhanced prompt (task.content.prompt) is written to
--ir-file, defaulting to prompt_ir.txt next to the input prompt.txt.
Every terminal run also writes a trace JSON — the exact submit request
(base64 data URLs elided to length markers) plus the terminal response —
to --trace, defaulting to h3_context_ir.json in the same directory (one
latest trace per case dir, overwritten each run); failure/cancelled
runs are traced too.

Env knobs (field names uppercased):
  MINIMAX_API_KEY     API bearer token                  (required)
  API_BASE            API root                          (https://api.minimax.cn)
  TASK_TYPE           t2va | i2va | r2va                (i2va)
  INPUT_DIR           case directory: prompt.txt + refs (task-aware default)
  DURATION            target video seconds, 4-15        (5)
  RATIO               adaptive|21:9|16:9|4:3|1:1|3:4|9:16 (task-aware default)
  POLL_INTERVAL_S     poll cadence, seconds             (5)
  POLL_TIMEOUT_S      give-up timeout, seconds          (900)

Exit code 0 = task succeeded; 1 = submission/HTTP error, task failure, or
timeout. The terminal-state JSON is printed last either way.

Docs:
  https://platform.minimaxi.com/docs/api-reference/video-generation-v2-h3-context-ir
      request schema: content type/role combinations per mode, media
      limits, ratio rules (request body <= 64 MB; large files need public
      URLs, not base64)
  https://platform.minimax.io/docs/api-reference/video-generation-v2-list
      V2 status enum: queued / running / succeeded / failed / cancelled
  https://github.com/MiniMax-AI/MiniMax-H3
      H3-Context-IR is hosted-API-only (not in the open release)
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

# V2 async task states, shared by /v2 video generation, h3_context_ir and
# video regeneration: queued -> running -> succeeded | failed | cancelled.
TERMINAL_OK = {"succeeded"}
TERMINAL_FAIL = {"failed", "cancelled"}

# Extension -> MIME for the reference modalities the h3_context_ir API
# accepts (stricter than generate.py's local-server sets: no .bmp/.mkv/
# .webm/.flac/.m4a here).
MIME_BY_EXT = {
    # image: JPG/JPEG/PNG/WEBP/HEIC/HEIF
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".webp": "image/webp", ".heic": "image/heic", ".heif": "image/heif",
    # video: MP4/MOV (multimodal reference only)
    ".mp4": "video/mp4", ".mov": "video/quicktime",
    # audio: WAV/MP3 (multimodal reference only)
    ".wav": "audio/wav", ".mp3": "audio/mpeg",
}
IMAGE_EXTS = {e for e, m in MIME_BY_EXT.items() if m.startswith("image/")}
VIDEO_EXTS = {e for e, m in MIME_BY_EXT.items() if m.startswith("video/")}
AUDIO_EXTS = {e for e, m in MIME_BY_EXT.items() if m.startswith("audio/")}

TASK_TYPES = {"t2va", "i2va", "r2va"}
RATIOS = {"adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"}
MAX_BODY_BYTES = 64 * 1024 * 1024   # API hard limit on the request body


@dataclass
class ContextIRConfig:
    """Request shape + API settings (mirrors the toolbox config layout).

    Empty input_dir / ratio mean "derive from task_type": input_dir from
    the per-task default case directory, ratio 16:9 for t2va (where the
    API forbids adaptive) and adaptive otherwise.
    """

    api_base: str = "https://api.minimax.cn"
    api_key: str = ""
    task_type: str = "i2va"
    input_dir: str = ""
    duration: int = 5
    ratio: str = ""
    poll_interval_s: float = 5.0
    poll_timeout_s: float = 900.0

    # Derived in __post_init__ from input_dir (same pattern as GenerateConfig).
    prompt_file: str = field(default="", init=False)  # input prompt.txt path
    prompt: str = field(default="", init=False)
    refs: list[dict] = field(default_factory=list, init=False)

    # Per-mode reference caps (enforced server-side by the API; checked
    # here for a clearer client-side error).
    REF_LIMITS = {"r2va": {"image": 9, "video": 3, "audio": 3}}

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("MINIMAX_API_KEY is required (bearer token)")
        if self.task_type not in TASK_TYPES:
            raise ValueError(f"TASK_TYPE must be one of {sorted(TASK_TYPES)}, "
                             f"got '{self.task_type}'")
        if not self.input_dir:
            self.input_dir = os.path.join(REPO_ROOT, "inputs", self.task_type)
        if not self.ratio:
            self.ratio = "16:9" if self.task_type == "t2va" else "adaptive"
        self._validate_request_params()
        self._load_inputs()

    def _validate_request_params(self) -> None:
        if not 4 <= self.duration <= 15:
            raise ValueError(f"DURATION must be in [4, 15], got "
                             f"{self.duration}")
        if self.ratio not in RATIOS:
            raise ValueError(f"RATIO must be one of {sorted(RATIOS)}, got "
                             f"'{self.ratio}'")
        if self.task_type == "t2va" and self.ratio == "adaptive":
            raise ValueError("t2va requires an explicit non-adaptive RATIO "
                             "(text-only requests have no image to adapt to)")
        if self.task_type == "i2va" and self.ratio != "adaptive":
            print(f"[h3-ir] NOTE: i2va ratio is decided by the frame image; "
                  f"ignoring RATIO={self.ratio}", file=sys.stderr)
            self.ratio = "adaptive"

    def _load_inputs(self) -> None:
        """Resolve prompt.txt + reference files from input_dir into refs."""
        self.prompt_file = os.path.join(self.input_dir, "prompt.txt")
        if not os.path.isfile(self.prompt_file):
            raise FileNotFoundError(
                f"{self.prompt_file} not found (check INPUT_DIR)")
        with open(self.prompt_file, encoding="utf-8") as fh:
            self.prompt = fh.read()
        # prompt_ir.txt (enhanced prompt) and *.json (context_ir_* traces,
        # custom --trace files) are this client's own output artifacts —
        # never references. .json is not an accepted modality anyway.
        paths = [os.path.join(self.input_dir, n)
                 for n in sorted(os.listdir(self.input_dir))
                 if n not in ("prompt.txt", "prompt_ir.txt")
                 and not n.startswith("README")
                 and os.path.splitext(n)[1].lower() != ".json"]
        kinds = [self._ref_kind(p) for p in paths]
        self._validate_refs(paths, kinds)
        self.refs = [
            {"kind": kind, "role": role, "path": path,
             "data_url": _data_url(path)}
            for path, kind, role in zip(paths, kinds, self._roles(kinds))
        ]

    def _ref_kind(self, path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        if ext in IMAGE_EXTS:
            return "image"
        if ext in VIDEO_EXTS:
            return "video"
        if ext in AUDIO_EXTS:
            return "audio"
        raise ValueError(f"unrecognized reference modality for '{path}' "
                         f"('{ext}' is not one the API accepts: image "
                         f"{sorted(IMAGE_EXTS)}, video {sorted(VIDEO_EXTS)}, "
                         f"audio {sorted(AUDIO_EXTS)})")

    def _validate_refs(self, paths: list[str], kinds: list[str]) -> None:
        if self.task_type == "t2va":
            if paths:
                raise ValueError(f"t2va takes no reference files, INPUT_DIR "
                                 f"holds {len(paths)}: {paths}")
        elif self.task_type == "i2va":
            bad = [p for p, k in zip(paths, kinds) if k != "image"]
            if bad:
                raise ValueError(f"i2va reference files must be images, got "
                                 f"other modalities: {bad}")
            if not paths:
                raise ValueError("i2va needs 1-2 frame images (first [+last]) "
                                 "in INPUT_DIR; for text-only use "
                                 "TASK_TYPE=t2va")
            if len(paths) > 2:
                raise ValueError(f"i2va takes at most 2 frame images (first "
                                 f"[+ last]); INPUT_DIR holds {len(paths)}")
        else:  # r2va
            if not paths:
                raise ValueError("r2va needs at least 1 reference file in "
                                 "INPUT_DIR; for text-only use TASK_TYPE=t2va")
            counts = {k: kinds.count(k) for k in ("image", "video", "audio")}
            for kind, limit in self.REF_LIMITS["r2va"].items():
                if counts[kind] > limit:
                    raise ValueError(f"r2va allows at most {limit} {kind} "
                                     f"reference(s), got {counts[kind]}")

    def _roles(self, kinds: list[str]) -> list[str]:
        """API role per reference, in upload order (after _validate_refs)."""
        if self.task_type == "i2va":
            return ["first_frame", "last_frame"][:len(kinds)]
        return [f"reference_{k}" for k in kinds]

    @property
    def ref_desc(self) -> str:
        n = len(self.refs)
        if self.task_type == "t2va":
            return "0 references (text-only)"
        if self.task_type == "i2va":
            return {1: "first frame", 2: "first + last frame"}[n]
        counts = {k: sum(r["kind"] == k for r in self.refs)
                  for k in ("image", "video", "audio")}
        return (f"{n} reference(s): {counts['image']} image(s) / "
                f"{counts['video']} video(s) / {counts['audio']} audio(s)")


class ContextIRClient:
    """Submit one H3-Context-IR task and block until it finishes."""

    def __init__(self, cfg: ContextIRConfig):
        self.cfg = cfg
        self.base = cfg.api_base.rstrip("/")
        self.last_payload: dict = {}   # submit()'s request body (for trace)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.cfg.api_key}"}

    def submit(self) -> str:
        """Create the task; returns its task_id."""
        cfg = self.cfg
        content: list[dict] = [{"type": "text", "text": cfg.prompt}]
        for r in cfg.refs:
            content.append({
                "type": f"{r['kind']}_url",
                f"{r['kind']}_url": {"url": r["data_url"]},
                "role": r["role"],
            })
        payload = {
            "model": "MiniMax-H3",
            "content": content,
            "duration": cfg.duration,
            "ratio": cfg.ratio,
        }
        self.last_payload = payload
        size = len(json.dumps(payload).encode("utf-8"))
        if size > MAX_BODY_BYTES:
            raise RuntimeError(
                f"request body is {size / 1e6:.0f} MB, over the API's 64 MB "
                f"limit; drop or shrink references (large files need public "
                f"URLs, which this client does not upload)")
        print(f"[h3-ir] {cfg.task_type} request ({cfg.ref_desc}), "
              f"duration={cfg.duration}s ratio={cfg.ratio}")
        resp = requests.post(f"{self.base}/v2/h3_context_ir",
                             json=payload,
                             headers={
                                 "Content-Type": "application/json",
                                 **self._headers(),
                             },
                             timeout=60)
        body = _json_or_text(resp)
        if resp.status_code != 200:
            raise RuntimeError(f"submit failed: HTTP {resp.status_code} {body}")
        task_id = (body.get("task_id") if isinstance(body, dict) else None) or ""
        if not task_id:
            raise RuntimeError(f"submit returned no task_id: {body}")
        print(f"[h3-ir] task submitted: {task_id}")
        return task_id

    def wait(self, task_id: str) -> dict:
        """Poll the query endpoint until a terminal state or timeout.

        The response nests the task under "task": {"status": ...}.
        """
        url = f"{self.base}/v2/query/video_generation/{task_id}"
        deadline = time.monotonic() + self.cfg.poll_timeout_s
        last: dict = {}
        while time.monotonic() < deadline:
            resp = requests.get(url, headers=self._headers(), timeout=30)
            last = _json_or_text(resp)
            status = _task_status(last)
            if status in TERMINAL_OK or status in TERMINAL_FAIL:
                print(f"[h3-ir] terminal state: {status}")
                return last
            print(f"[h3-ir] {time.strftime('%H:%M:%S')} status={status or '?'} "
                  f"(polling every {self.cfg.poll_interval_s:g}s)")
            time.sleep(self.cfg.poll_interval_s)
        raise TimeoutError(f"task {task_id} not terminal after "
                           f"{self.cfg.poll_timeout_s:g}s; last response: {last}")

    def run(self, ir_file: str | None = None,
            trace_file: str | None = None) -> dict:
        """Submit + wait; returns the terminal response dict.

        Prints the run time: local wall clock (submit -> terminal, so up
        to one poll interval longer than the task itself) plus the
        server-side created_at -> updated_at span when present. Always
        writes a trace JSON ({"request": ..., "response": ...}; data URLs
        elided) to trace_file, defaulting to h3_context_ir.json next to
        cfg.prompt_file (overwritten each run). On success, additionally
        saves the enhanced prompt (task.content.prompt) to ir_file,
        defaulting to prompt_ir.txt next to cfg.prompt_file.
        """
        start = time.monotonic()
        result = self.wait(self.submit())
        elapsed = time.monotonic() - start
        msg = f"[h3-ir] run time: {elapsed:.1f}s (submit -> terminal)"
        server = _task_duration(result)
        if server is not None:
            msg += f"; {server}s server-side (created_at -> updated_at)"
        print(msg)
        self._write_trace(result, trace_file)
        if _task_status(result) not in TERMINAL_OK:
            return result
        path = ir_file or os.path.join(
            os.path.dirname(self.cfg.prompt_file) or ".", "prompt_ir.txt")
        prompt = _enhanced_prompt(result)
        if not prompt:
            print(f"[h3-ir] WARNING: no task.content.prompt in result; "
                  f"nothing written to {path}")
            return result
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(prompt)
        print(f"[h3-ir] enhanced prompt saved: {path} ({len(prompt)} chars)")
        return result

    def _write_trace(self, result: dict, trace_file: str | None) -> None:
        """Persist the submit request + terminal response as one JSON.

        Written on failure/cancel too (a TimeoutError in wait() raises
        past run(), so a timed-out run leaves no trace).
        """
        path = trace_file or os.path.join(
            os.path.dirname(self.cfg.prompt_file) or ".",
            "h3_context_ir.json")
        doc = {"request": _elide_data_urls(self.last_payload),
               "response": result}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, ensure_ascii=False)
        print(f"[h3-ir] trace saved: {path}")


def _json_or_text(resp: requests.Response):
    try:
        return resp.json()
    except ValueError:
        return {"raw": resp.text, "status_code": resp.status_code}


def _task_status(body) -> str:
    """Lowercased task status from a query response ('' when absent).

    Observed shape: {"task": {"status": "succeeded", ...}}.
    """
    if not isinstance(body, dict):
        return ""
    task = body.get("task")
    if isinstance(task, dict):
        return str(task.get("status", "") or "").lower()
    return ""


def _enhanced_prompt(body) -> str:
    """task.content.prompt from a terminal response ('' when absent)."""
    task = body.get("task") if isinstance(body, dict) else None
    content = task.get("content") if isinstance(task, dict) else None
    if isinstance(content, dict):
        return str(content.get("prompt", "") or "")
    return ""


def _task_duration(body) -> int | None:
    """task.updated_at - task.created_at (unix seconds), None when absent."""
    task = body.get("task") if isinstance(body, dict) else None
    if not isinstance(task, dict):
        return None
    try:
        return int(task["updated_at"]) - int(task["created_at"])
    except (KeyError, TypeError, ValueError):
        return None


def _data_url(path: str) -> str:
    """Local file -> base64 data URL (the API accepts link or base64)."""
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    mime = MIME_BY_EXT[os.path.splitext(path)[1].lower()]
    return f"data:{mime};base64,{b64}"


def _elide_data_urls(obj):
    """Deep copy with inline base64 data URLs replaced by length markers.

    The media itself lives in INPUT_DIR; eliding keeps the request trace
    small and diffable.
    """
    if isinstance(obj, dict):
        return {k: _elide_data_urls(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_elide_data_urls(v) for v in obj]
    if isinstance(obj, str) and obj.startswith("data:") and len(obj) > 128:
        return f"{obj[:48]}...<{len(obj)} chars, elided>"
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Submit one MiniMax H3-Context-IR task and wait for it.")
    parser.add_argument("--task-type", default=None, metavar="TYPE",
                        choices=sorted(TASK_TYPES),
                        help="request mode (default: env TASK_TYPE, or i2va)")
    parser.add_argument("--input-dir", default=None, metavar="DIR",
                        help="case directory holding prompt.txt + mode-"
                             "dependent reference files (default: env "
                             "INPUT_DIR, or inputs/<task_type>)")
    parser.add_argument("--duration", type=int, default=None, metavar="S",
                        help="target video seconds, 4-15 (default: env "
                             "DURATION, or 5)")
    parser.add_argument("--ratio", default=None, metavar="W:H",
                        help="target aspect ratio (default: env RATIO, or "
                             "task-aware: 16:9 for t2va, adaptive otherwise)")
    parser.add_argument("--ir-file", default=None, metavar="FILE",
                        help="where to save the enhanced prompt on success "
                             "(default: prompt_ir.txt next to the input "
                             "prompt.txt)")
    parser.add_argument("--trace", default=None, metavar="FILE",
                        help="where to save the request+response trace JSON "
                             "(default: h3_context_ir.json next to the "
                             "input prompt.txt, overwritten each run)")
    args = parser.parse_args()

    def env(name: str, default: Any, cast: Callable[[str], Any] = str) -> Any:
        raw = os.environ.get(name, "")
        return cast(raw) if raw else default

    try:
        # CLI > env > task-aware default: absent flags fall back to their
        # env twins; an empty input_dir/ratio then derives from task_type
        # in __post_init__ — one construction, so validation only ever
        # sees the final values.
        cfg = ContextIRConfig(
            api_base=env("API_BASE", "https://api.minimax.cn"),
            api_key=env("MINIMAX_API_KEY", ""),
            task_type=args.task_type or env("TASK_TYPE", "i2va"),
            input_dir=args.input_dir or env("INPUT_DIR", ""),
            duration=args.duration if args.duration is not None
                     else env("DURATION", 5, int),
            ratio=args.ratio or env("RATIO", ""),
            poll_interval_s=env("POLL_INTERVAL_S", 5.0, float),
            poll_timeout_s=env("POLL_TIMEOUT_S", 900.0, float),
        )
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        result = ContextIRClient(cfg).run(ir_file=args.ir_file,
                                          trace_file=args.trace)
    except (requests.RequestException, RuntimeError, TimeoutError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if _task_status(result) in TERMINAL_OK else 1


if __name__ == "__main__":
    sys.exit(main())
