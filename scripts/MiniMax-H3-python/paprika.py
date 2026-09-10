#!/usr/bin/env python3
"""Reproduce a Paprika MiniMax-H3 task locally: manifest + assets + video.

Python port of tools/RadiantFrame/reproduce_task.sh (drop-in equivalent):
PAPRIKA_KEY + a task id -> one self-contained reproduction directory,

    inputs/paprika-repro-task_<id>/
      task.json                    reproduction manifest (verbatim)
      prompt.txt                   .prompt
      parameters.json              .parameters
      context-ir-prompt.txt        .context_ir_optimized_prompt (if present)
      inputs/NN_<role>_<id>.<ext>  input/reference assets, manifest order
      outputs/video.mp4            the generated video (when available)

Usage:
    PAPRIKA_KEY=... python paprika.py --task-id TASK_ID [--output-dir DIR]
    PAPRIKA_KEY=... TASK_ID=task_... python paprika.py

Env knobs (field names uppercased):
  PAPRIKA_KEY        API key                           (required)
  PAPRIKA_BASE_URL   API origin                        (http://8.130.171.112)
  TASK_ID            task id when no positional arg is given
  OUTPUT_DIR         export dir when no positional arg is given
                     (default ./inputs/paprika-repro-<task_id>)
  REQUEST_TIMEOUT    per-request read timeout, seconds (300)

The key may be a project API Key for a task in that project, or the
system Master Key for a read-only cross-project diagnostic export.
The key is never printed or written into the export directory; files
land with 0600 perms and directories with 0700 (bash umask-077 parity).

Exit code 0 = exported; 1 = bad args, HTTP error, or invalid manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Callable

import requests

DEFAULT_BASE_URL = "http://8.130.171.112"
TASK_ID_RE = re.compile(r"^task_[A-Za-z0-9]+$")
ORIGIN_RE = re.compile(r"^https?://[^/]+(?::\d+)?$")
ASSET_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
# Manifest asset extension by declared mime type ("" -> keep as downloaded,
# bash parity: unknown mimes get no extension).
EXT_BY_MIME = {
    "image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp",
    "video/mp4": ".mp4", "audio/mpeg": ".mp3",
    "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/mp4": ".m4a",
}


@dataclass
class PaprikaConfig:
    """One export: where from (base_url + task) and where to (output_dir)."""

    api_key: str
    task_id: str
    base_url: str = DEFAULT_BASE_URL
    output_dir: str = ""
    request_timeout: float = 300.0

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("PAPRIKA_KEY is required")
        if not TASK_ID_RE.match(self.task_id):
            raise ValueError(f"invalid task id: {self.task_id!r} "
                             f"(expected task_<alphanumerics>)")
        self.base_url = self.base_url.rstrip("/")
        if not ORIGIN_RE.match(self.base_url):
            raise ValueError(f"PAPRIKA_BASE_URL must be an http(s) origin "
                             f"without a path, got {self.base_url!r}")
        if self.base_url.startswith("http://"):
            print("[paprika] WARNING: using the internal plaintext HTTP "
                  "test endpoint", file=sys.stderr)
        if not self.output_dir:
            self.output_dir = f"./inputs/paprika-repro-{self.task_id}"


class PaprikaClient:
    """Fetch one task's reproduction manifest and its artifacts."""

    def __init__(self, cfg: PaprikaConfig):
        self.cfg = cfg

    # -- transport ------------------------------------------------------

    def _key_headers(self) -> dict[str, str]:
        return {"Authorization": f"Key {self.cfg.api_key}"}

    def _get(self, url: str, *, auth: bool, dest: str | None = None):
        """GET url -> response text, or stream the body to dest.

        Authed calls never follow redirects (curl-without-L parity: a
        redirect on an API path should fail loudly, and it also keeps the
        Key header from ever traveling cross-host); source_url fallbacks
        follow redirects with no auth, mirroring the bash curl -L.
        """
        resp = requests.get(
            url,
            headers=self._key_headers() if auth else {},
            timeout=self.cfg.request_timeout,
            stream=dest is not None,
            allow_redirects=not auth,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"GET {url} failed: HTTP {resp.status_code} "
                               f"{resp.text[:2000]}")
        if dest is None:
            return resp.text
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
        os.chmod(dest, 0o600)
        return None

    def _asset_url(self, asset_id: str) -> str:
        return (f"{self.cfg.base_url}/v1/minimax-h3/requests/"
                f"{self.cfg.task_id}/input-assets/{asset_id}")

    # -- export steps ---------------------------------------------------

    def fetch_manifest(self) -> dict:
        """GET the reproduction manifest; validate it matches the task."""
        url = (f"{self.cfg.base_url}/v1/minimax-h3/requests/"
               f"{self.cfg.task_id}/reproduction")
        text = self._get(url, auth=True)
        try:
            manifest = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"manifest is not valid JSON: {exc}") from exc
        if not isinstance(manifest, dict) \
                or manifest.get("request_id") != self.cfg.task_id:
            raise RuntimeError(
                "API returned an invalid reproduction manifest "
                f"(request_id != {self.cfg.task_id}): {text[:2000]}")
        return manifest

    def export(self) -> dict:
        """Write the whole reproduction directory; returns the manifest."""
        cfg = self.cfg
        out = cfg.output_dir
        for sub in ("inputs", "outputs"):
            os.makedirs(os.path.join(out, sub), exist_ok=True)
        os.chmod(out, 0o700)
        for sub in ("inputs", "outputs"):
            os.chmod(os.path.join(out, sub), 0o700)

        manifest = self.fetch_manifest()
        self._write_file(json.dumps(manifest, ensure_ascii=False),
                         os.path.join(out, "task.json"))
        self._write_file(str(manifest.get("prompt") or ""),
                         os.path.join(out, "prompt.txt"))
        self._write_file(json.dumps(manifest.get("parameters"),
                                    indent=2, ensure_ascii=False),
                         os.path.join(out, "parameters.json"))
        ir_prompt = manifest.get("context_ir_optimized_prompt")
        if ir_prompt is not None:
            self._write_file(str(ir_prompt),
                             os.path.join(out, "context-ir-prompt.txt"))

        asset_count = self._download_assets(manifest, os.path.join(out, "inputs"))
        video_path = self._download_video(manifest, os.path.join(out, "outputs"))

        status = manifest.get("status")
        capability = manifest.get("capability")
        print(f"[paprika] exported {cfg.task_id} ({capability}, {status}) "
              f"to {out}")
        print(f"[paprika] prompt: {os.path.join(out, 'prompt.txt')}")
        print(f"[paprika] manifest: {os.path.join(out, 'task.json')}")
        print(f"[paprika] input assets: {asset_count}")
        if video_path:
            print(f"[paprika] video: {video_path}")
        else:
            print(f"[paprika] video: not available for task status {status}")
        return manifest

    def _write_file(self, text: str, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.chmod(path, 0o600)

    def _download_assets(self, manifest: dict, inputs_dir: str) -> int:
        """Save every unique input/reference asset, manifest order.

        download_url present -> the authed API asset endpoint; else
        source_url -> direct fetch (no auth, redirects followed); else
        the asset has no downloadable copy (warn).
        """
        seen: set[str] = set()
        assets = []
        for key in ("input_assets", "reference_assets"):
            for asset in manifest.get(key) or []:
                aid = asset.get("asset_id")
                if aid and aid not in seen:
                    seen.add(aid)
                    assets.append(asset)
        for n, asset in enumerate(assets, 1):
            aid = asset["asset_id"]
            role = _sanitize(str(asset.get("role") or "asset"),
                             keep_upper=False)
            ext = EXT_BY_MIME.get(asset.get("mime_type") or "", "")
            dest = os.path.join(inputs_dir, f"{n:02d}_{role}_{aid}{ext}")
            if asset.get("download_url"):
                if not ASSET_ID_RE.match(aid):
                    raise RuntimeError(f"invalid asset id in manifest: {aid}")
                self._get(self._asset_url(aid), auth=True, dest=dest)
            elif asset.get("source_url"):
                self._get(asset["source_url"], auth=False, dest=dest)
            else:
                print(f"[paprika] WARNING: asset {aid} has no downloadable "
                      f"copy", file=sys.stderr)
        return len(assets)

    def _download_video(self, manifest: dict, outputs_dir: str) -> str | None:
        """Save the generated video when the task has one."""
        dest = os.path.join(outputs_dir, "video.mp4")
        if manifest.get("video_download_url"):
            self._get(f"{self.cfg.base_url}/v1/minimax-h3/requests/"
                      f"{self.cfg.task_id}/video", auth=True, dest=dest)
            return dest
        if manifest.get("video_source_url"):
            self._get(manifest["video_source_url"], auth=False, dest=dest)
            return dest
        return None


def _sanitize(text: str, keep_upper: bool = True) -> str:
    """Bash tr parity: non [A-Za-z0-9._-] -> '_' (role also lowercased)."""
    out = []
    for ch in text:
        if ch.isalnum() or ch in "._-":
            out.append(ch if keep_upper else ch.lower())
        else:
            out.append("_")
    return "".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce a Paprika MiniMax-H3 task locally "
                    "(manifest + assets + video).")
    parser.add_argument("--task-id", default=None, metavar="TASK_ID",
                        help="task_<id> to export (default: env TASK_ID)")
    parser.add_argument("--output-dir", default=None, metavar="DIR",
                        help="export directory (default: env OUTPUT_DIR, or "
                             "./inputs/paprika-repro-<task_id>)")
    args = parser.parse_args()

    def env(name: str, default: Any, cast: Callable[[str], Any] = str) -> Any:
        raw = os.environ.get(name, "")
        return cast(raw) if raw else default

    try:
        # CLI > env, one construction; REQUEST_ID is the bash-era fallback.
        cfg = PaprikaConfig(
            api_key=env("PAPRIKA_KEY", ""),
            task_id=(args.task_id or env("TASK_ID", "")
                     or os.environ.get("REQUEST_ID", "")),
            base_url=env("PAPRIKA_BASE_URL", DEFAULT_BASE_URL),
            output_dir=args.output_dir or env("OUTPUT_DIR", ""),
            request_timeout=env("REQUEST_TIMEOUT", 300.0, float),
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        PaprikaClient(cfg).export()
    except (requests.RequestException, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
