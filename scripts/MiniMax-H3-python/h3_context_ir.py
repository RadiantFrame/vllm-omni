#!/usr/bin/env python3
"""One-shot MiniMax H3-Context-IR client: submit -> poll -> print result.

Wraps the two-step MiniMax API (create a H3-Context-IR task, then poll the
video-generation query endpoint) into a single blocking call: the script
exits when the task reaches a terminal state and prints the final result
JSON. Unlike video generation, the result is text (modality: "text"): the
enhanced prompt / context IR lives in task.content.prompt, ready to feed
the follow-up video-generation call.

Usage:
    MINIMAX_API_KEY=... python h3_context_ir.py [--payload FILE]
                                               [--ir-file FILE] [--trace FILE]

The single input is a payload.json — the request body, with local files:

  {
    "model": "MiniMax-H3",
    "content": [
      {"type": "text", "text": "prompt.txt"},
      {"type": "image_url",
       "image_url": {"url": "references/01_image.jpg"},
       "role": "first_frame"}
    ],
    "duration": 5,
    "ratio": "adaptive"
  }

The text prompt and media urls are file paths RELATIVE to the payload
file's directory (any nesting works); the client reads each file and
sends the prompt verbatim and the media as base64 data URLs (http(s)
URLs are not supported — download the file first). roles and the docs'
mode rules, mirrored here:

  text-only    no media items; ratio must be explicit (not adaptive)
  frame        1-2 image_url items, roles first_frame [+ last_frame];
               ratio is decided by the frame (adaptive)
  reference    image/video/audio items, roles reference_<kind>; the API
               caps them at 9 images / 3 videos / 3 audios and validates
               the combination server-side (clear 2013-style errors)

duration (4-15) and ratio may be omitted (the API defaults them).

On success the enhanced prompt (task.content.prompt) is written to
--ir-file, defaulting to h3_context_ir_prompt.txt next to the payload
file (generate.py's USE_CONTEXT_IR_PROMPT reads it from there). Every
terminal run also writes a trace JSON — the exact submit request (base64
data URLs elided to length markers) plus the terminal response — to
--trace, defaulting to h3_context_ir.json in the same directory (one
latest trace per case dir, overwritten each run); failure/cancelled runs
are traced too. The payload file thus doubles as the input record.

Env knobs (field names uppercased):
  MINIMAX_API_KEY     API bearer token                  (required)
  API_BASE            API root                          (https://api.minimax.cn)
  PAYLOAD_FILE        the payload.json to submit        (inputs/i2va/payload.json)
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
DEFAULT_PAYLOAD = os.path.join(REPO_ROOT, "inputs", "i2va", "payload.json")

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
    # audio: WAV/MP3 (multimodal reference only). audio/mp3, not the
    # RFC-correct audio/mpeg: the API validates the data-URL subtype as
    # the file format and rejects ".mpeg" (error 2013).
    ".wav": "audio/wav", ".mp3": "audio/mp3",
}
# content item types that carry a media file (their inner dict key equals
# the type, e.g. {"type": "image_url", "image_url": {"url": ...}}).
MEDIA_URL_TYPES = {"image_url", "video_url", "audio_url"}

RATIOS = {"adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"}
MAX_BODY_BYTES = 64 * 1024 * 1024   # API hard limit on the request body


@dataclass
class ContextIRConfig:
    """One payload.json + API settings.

    __post_init__ loads and validates the payload, resolving each media
    url to an absolute path (checked to exist, with an accepted
    extension) for submit()'s base64 conversion.
    """

    api_base: str = "https://api.minimax.cn"
    api_key: str = ""
    payload_file: str = ""   # --payload: the request body, local-file urls
    poll_interval_s: float = 5.0
    poll_timeout_s: float = 900.0

    # Derived in __post_init__: verbatim payload, its directory (anchors
    # the trace / enhanced-prompt defaults), and content-index -> absolute
    # file paths for submit()'s path -> content swaps (text file read,
    # media file base64'd).
    payload: dict = field(default_factory=dict, init=False)
    payload_dir: str = field(default="", init=False)
    text_paths: dict[int, str] = field(default_factory=dict, init=False)
    media_paths: dict[int, str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("MINIMAX_API_KEY is required (bearer token)")
        if not self.payload_file:
            self.payload_file = DEFAULT_PAYLOAD
        self.payload_file = os.path.abspath(self.payload_file)
        self.payload_dir = os.path.dirname(self.payload_file)
        self._load_payload()

    def _load_payload(self) -> None:
        """Read the payload.json, sanity-check it, resolve media paths.

        The body is submitted verbatim except for the media url ->
        base64 data-URL swap; these checks only mirror the API's own
        constraints (model name, content shape, duration range, ratio
        enum, accepted file formats) for a clearer client-side error.
        Mode-specific rules (roles, media combinations) are validated
        server-side. duration and ratio may be omitted (the API defaults
        them).
        """
        with open(self.payload_file, encoding="utf-8") as fh:
            try:
                payload = json.load(fh)
            except ValueError as exc:
                raise ValueError(f"{self.payload_file} is not valid JSON: "
                                 f"{exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{self.payload_file} must hold a JSON object "
                             f"(the request body), got "
                             f"{type(payload).__name__}")
        if payload.get("model") != "MiniMax-H3":
            raise ValueError(f"{self.payload_file}: model must be "
                             f"'MiniMax-H3', got {payload.get('model')!r}")
        content = payload.get("content")
        if not isinstance(content, list) or not content:
            raise ValueError(f"{self.payload_file}: needs a non-empty "
                             f"'content' array")
        for i, item in enumerate(content):
            self._validate_item(i, item)
        if "duration" in payload and not 4 <= payload["duration"] <= 15:
            raise ValueError(f"{self.payload_file}: duration must be in "
                             f"[4, 15], got {payload['duration']}")
        if "ratio" in payload and payload["ratio"] not in RATIOS:
            raise ValueError(f"{self.payload_file}: ratio must be one of "
                             f"{sorted(RATIOS)}, got {payload['ratio']!r}")
        self.payload = payload

    def _validate_item(self, i: int, item) -> None:
        """Check one content item; record media urls in media_paths."""
        where = f"{self.payload_file}: content[{i}]"
        if not isinstance(item, dict) or "type" not in item:
            raise ValueError(f"{where} must be an object with a 'type'")
        kind = item["type"]
        if kind == "text":
            text = item.get("text")
            if not isinstance(text, str) or not text:
                raise ValueError(f"{where} text item needs a string 'text' "
                                 f"(a prompt file path)")
            path = os.path.normpath(os.path.join(self.payload_dir, text))
            if not os.path.isfile(path):
                raise ValueError(f"{where} text '{text}' -> {path} not found "
                                 f"(text values are file paths relative to "
                                 f"the payload file)")
            self.text_paths[i] = path
            return
        if kind not in MEDIA_URL_TYPES:
            raise ValueError(f"{where} type must be 'text' or one of "
                             f"{sorted(MEDIA_URL_TYPES)}, got {kind!r}")
        holder = item.get(kind)
        url = holder.get("url") if isinstance(holder, dict) else None
        if not isinstance(url, str) or not url:
            raise ValueError(f"{where} {kind} item needs "
                             f"{{'{kind}': {{'url': ...}}}} with a file path")
        path = os.path.normpath(os.path.join(self.payload_dir, url))
        if not os.path.isfile(path):
            raise ValueError(f"{where} url '{url}' -> {path} not found "
                             f"(paths are relative to the payload file)")
        ext = os.path.splitext(path)[1].lower()
        mime = MIME_BY_EXT.get(ext)
        if mime is None or not mime.startswith(f"{kind.removesuffix('_url')}/"):
            accepted = {e: m for e, m in MIME_BY_EXT.items()
                        if m.startswith(f"{kind.removesuffix('_url')}/")}
            raise ValueError(f"{where} url '{url}': '{ext}' does not match "
                             f"{kind} (accepted: {sorted(accepted)})")
        self.media_paths[i] = path

    @property
    def ref_desc(self) -> str:
        """One-line media summary of the payload, for submit()'s print."""
        media = [item for item in self.payload.get("content", [])
                 if item.get("type") != "text"]
        items = ", ".join(
            f"{i.get('type', '?')}"
            + (f"({i['role']})" if i.get("role") else "")
            for i in media) or "text only"
        return items


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
        payload = cfg.payload
        if cfg.media_paths or cfg.text_paths:
            # Copy-on-write swap of local file paths -> file contents
            # (prompt text verbatim, media base64'd); cfg.payload (the
            # on-disk record) keeps the relative paths.
            content = []
            for i, item in enumerate(payload["content"]):
                if i in cfg.text_paths:
                    item = {**item, "text": _read_text(cfg.text_paths[i])}
                elif i in cfg.media_paths:
                    item = dict(item)
                    item[item["type"]] = {**item[item["type"]],
                                          "url": _data_url(cfg.media_paths[i])}
                content.append(item)
            payload = {**payload, "content": content}
        self.last_payload = payload
        size = len(json.dumps(payload).encode("utf-8"))
        if size > MAX_BODY_BYTES:
            raise RuntimeError(
                f"request body is {size / 1e6:.0f} MB, over the API's 64 MB "
                f"limit; drop or shrink media files (base64 inflates them "
                f"~33%; public URLs are not supported by this client)")
        print(f"[h3-ir] payload request from {cfg.payload_file} "
              f"({cfg.ref_desc})")
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
        the payload file (overwritten each run). On success, additionally
        saves the enhanced prompt (task.content.prompt) to ir_file,
        defaulting to h3_context_ir_prompt.txt next to the payload file.
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
        path = ir_file or os.path.join(self.cfg.payload_dir,
                                       "h3_context_ir_prompt.txt")
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
        path = trace_file or os.path.join(self.cfg.payload_dir,
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


def _read_text(path: str) -> str:
    """Local file -> prompt text (utf-8)."""
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _data_url(path: str) -> str:
    """Local file -> base64 data URL (the API accepts link or base64)."""
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    mime = MIME_BY_EXT[os.path.splitext(path)[1].lower()]
    return f"data:{mime};base64,{b64}"


def _elide_data_urls(obj):
    """Deep copy with inline base64 data URLs replaced by length markers.

    The media itself lives next to the payload file; eliding keeps the
    request trace small and diffable.
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
    parser.add_argument("--payload", default=None, metavar="FILE",
                        help="payload.json: the request body (model/content/"
                             "duration/ratio, roles inline; the text prompt "
                             "and media urls are file paths relative to this "
                             "file, sent verbatim / as base64 data URLs) "
                             "(default: env PAYLOAD_FILE, or "
                             "inputs/i2va/payload.json)")
    parser.add_argument("--ir-file", default=None, metavar="FILE",
                        help="where to save the enhanced prompt on success "
                             "(default: h3_context_ir_prompt.txt next to the "
                             "payload file)")
    parser.add_argument("--trace", default=None, metavar="FILE",
                        help="where to save the request+response trace JSON "
                             "(default: h3_context_ir.json next to the "
                             "payload file, overwritten each run)")
    args = parser.parse_args()

    def env(name: str, default: Any, cast: Callable[[str], Any] = str) -> Any:
        raw = os.environ.get(name, "")
        return cast(raw) if raw else default

    try:
        # CLI > env > default: one construction, so _load_payload's
        # validation only ever sees the final payload_file.
        cfg = ContextIRConfig(
            api_base=env("API_BASE", "https://api.minimax.cn"),
            api_key=env("MINIMAX_API_KEY", ""),
            payload_file=args.payload or env("PAYLOAD_FILE", ""),
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
