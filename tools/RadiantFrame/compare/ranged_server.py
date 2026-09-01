#!/usr/bin/env python3
"""Minimal local HTTP server with Range support for the compare player.

python3 -m http.server ignores Range headers (answers 200 with the full
body), which makes Chrome disable/fail <video> seeking — frame stepping
resets to 0 and playback controls die. This handler answers Range
requests with 206/Content-Range so seeking works.

Usage: python3 ranged_server.py PORT [ROOT]   (defaults: 8000 /)
Bound to 127.0.0.1 only.
"""
import os
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class RangeHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        if not os.path.exists(path) or not os.path.isfile(path):
            self.send_error(404)
            return None

        size = os.path.getsize(path)
        range_header = self.headers.get("Range")
        if range_header:
            m = re.match(r"bytes=(\d*)-(\d*)$", range_header.strip())
            if m and (m.group(1) or m.group(2)):
                start = int(m.group(1)) if m.group(1) else None
                end = int(m.group(2)) if m.group(2) else None
                if start is None:  # suffix range: bytes=-N
                    start = max(0, size - (end or 0))
                    end = size - 1
                elif end is None or end >= size:
                    end = size - 1
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return None
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", str(end - start + 1))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Type", self.guess_type(path))
                self.end_headers()
                return _SliceReader(path, start, end)

        self.send_response(200)
        self.send_header("Content-Length", str(size))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Type", self.guess_type(path))
        self.end_headers()
        return open(path, "rb")


class _SliceReader:
    """File-like object serving only [start, end] of a file."""

    def __init__(self, path, start, end):
        self.f = open(path, "rb")
        self.f.seek(start)
        self.remaining = end - start + 1

    def read(self, n=-1):
        if n < 0 or n > self.remaining:
            n = self.remaining
        data = self.f.read(n)
        self.remaining -= len(data)
        return data

    def close(self):
        self.f.close()


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    root = sys.argv[2] if len(sys.argv) > 2 else "/"
    os.chdir(root)
    server = ThreadingHTTPServer(("127.0.0.1", port), RangeHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
