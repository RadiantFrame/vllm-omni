#!/usr/bin/env python3
"""One-shot MiniMax H3-Context-IR client: submit -> poll -> print result.

Wraps the two-step MiniMax API (create a H3-Context-IR task, then poll the
video-generation query endpoint) into a single blocking call: the script
exits when the task reaches a terminal state and prints the final result
JSON (including the generated video's download URL).

Usage:
    MINIMAX_API_KEY=... python minimax_h3.py [--input-dir DIR]

Inputs come from INPUT_DIR (same convention as generate.py): a case
directory holding prompt.txt plus 0-1 reference image (sorted filename
order); the local image is uploaded as a base64 data URL. --input-dir
overrides the INPUT_DIR env knob.

Env knobs (field names uppercased):
  MINIMAX_API_KEY     API bearer token                  (required)
  API_BASE            API root                          (https://api.minimax.cn)
  INPUT_DIR           case directory: prompt.txt + 0-1 image
                      (default <repo>/inputs/t2v)
  DURATION            video seconds                     (5)
  RATIO               aspect ratio                      (adaptive)
  POLL_INTERVAL_S     poll cadence, seconds             (5)
  POLL_TIMEOUT_S      give-up timeout, seconds          (900)

Exit code 0 = task succeeded; 1 = submission/HTTP error, task failure, or
timeout. The terminal-state JSON is printed last either way.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import json
import mimetypes
import os
import sys
import time
from dataclasses import dataclass, field

import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

TERMINAL_OK = {"success", "succeed", "succeeded", "done", "success_finish"}
TERMINAL_FAIL = {"fail", "failed", "error", "fail_finish", "cancelled"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class ContextIRConfig:
    """Request shape + API settings (mirrors the toolbox config layout)."""

    api_base: str = "https://api.minimax.cn"
    api_key: str = ""
    input_dir: str = os.path.join(REPO_ROOT, "inputs", "t2v")
    duration: int = 5
    ratio: str = "adaptive"
    poll_interval_s: float = 5.0
    poll_timeout_s: float = 900.0

    # Derived in __post_init__ from input_dir (same pattern as GenerateConfig).
    prompt: str = field(default="", init=False)
    image_ref: str = field(default="", init=False)   # data URL, "" = text-only

    @classmethod
    def from_env(cls) -> "ContextIRConfig":
        def env(name: str, default, cast=str):
            raw = os.environ.get(name, "")
            return cast(raw) if raw else default

        return cls(
            api_base=env("API_BASE", "https://api.minimax.cn"),
            api_key=env("MINIMAX_API_KEY", ""),
            input_dir=env("INPUT_DIR", os.path.join(REPO_ROOT, "inputs", "t2v")),
            duration=env("DURATION", 5, int),
            ratio=env("RATIO", "adaptive"),
            poll_interval_s=env("POLL_INTERVAL_S", 5.0, float),
            poll_timeout_s=env("POLL_TIMEOUT_S", 900.0, float),
        )

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("MINIMAX_API_KEY is required (bearer token)")
        prompt_file = os.path.join(self.input_dir, "prompt.txt")
        if not os.path.isfile(prompt_file):
            raise FileNotFoundError(f"{prompt_file} not found (check INPUT_DIR)")
        with open(prompt_file, encoding="utf-8") as fh:
            self.prompt = fh.read()
        refs = [os.path.join(self.input_dir, n)
                for n in sorted(os.listdir(self.input_dir))
                if n != "prompt.txt" and not n.startswith("README")]
        for r in refs:
            if os.path.splitext(r)[1].lower() not in IMAGE_EXTS:
                raise ValueError(f"reference must be an image, got: {r}")
        if len(refs) > 1:
            raise ValueError(f"H3-Context-IR takes at most 1 first_frame "
                             f"image, INPUT_DIR holds {len(refs)}: {refs}")
        if refs:
            # Local image -> base64 data URL (the API accepts link or
            # base64 for image_url).
            with open(refs[0], "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode("ascii")
            mime = mimetypes.guess_type(refs[0])[0] or "image/png"
            self.image_ref = f"data:{mime};base64,{b64}"
        else:
            self.image_ref = ""


class ContextIRClient:
    """Submit one H3-Context-IR task and block until it finishes."""

    def __init__(self, cfg: ContextIRConfig):
        self.cfg = cfg
        self.base = cfg.api_base.rstrip("/")

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.cfg.api_key}"}

    def submit(self) -> str:
        """Create the task; returns its task_id."""
        content: list[dict] = [{"type": "text", "text": self.cfg.prompt}]
        if self.cfg.image_ref:
            content.append({
                "type": "image_url",
                "image_url": {"url": self.cfg.image_ref},
                "role": "first_frame",
            })
        payload = {
            "model": "MiniMax-H3",
            "content": content,
            "duration": self.cfg.duration,
            "ratio": self.cfg.ratio,
        }
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

    def run(self) -> dict:
        """Submit + wait; returns the terminal response dict."""
        return self.wait(self.submit())


def _json_or_text(resp: requests.Response):
    try:
        return resp.json()
    except ValueError:
        return {"raw": resp.text, "status_code": resp.status_code}


def _task_status(body) -> str:
    """Lowercased task status from a query response ('' when absent).

    Accepts both the observed nested shape {"task": {"status": ...}} and a
    flat {"status": ...} just in case.
    """
    if not isinstance(body, dict):
        return ""
    task = body.get("task")
    if isinstance(task, dict) and task.get("status") is not None:
        return str(task["status"]).lower()
    return str(body.get("status", "") or "").lower()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Submit one MiniMax H3-Context-IR task and wait for it.")
    parser.add_argument("--input-dir", default=None, metavar="DIR",
                        help="case directory holding prompt.txt + 0-1 image "
                             "(default: env INPUT_DIR, or <repo>/inputs/t2v)")
    args = parser.parse_args()

    try:
        cfg = ContextIRConfig.from_env()
        if args.input_dir:
            # replace() re-runs __post_init__, re-deriving prompt/image_ref.
            cfg = dataclasses.replace(cfg, input_dir=args.input_dir)
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        result = ContextIRClient(cfg).run()
    except (requests.RequestException, RuntimeError, TimeoutError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if _task_status(result) in TERMINAL_OK else 1


if __name__ == "__main__":
    sys.exit(main())
