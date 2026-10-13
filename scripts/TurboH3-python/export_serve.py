#!/usr/bin/env python3
"""TurboH3 export wrapper — reuses MiniMax-H3-python's export_serve.

Same relationship as pipeline.py: the rendering lives on the shared
Config classes, so this is a thin import wrapper. The default output
directory is derived purely from the config path — its configs*
segment swapped for paprika* in place, so configs/fl2va/rtx5090
exports to scripts/TurboH3-python/paprika/fl2va/rtx5090/ and
configs_4steps/... to paprika_4steps/... (tier suffix kept).

Usage (from the repo root):
    python scripts/TurboH3-python/export_serve.py \
        --config scripts/TurboH3-python/configs/fl2va/rtx5090/config.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "MiniMax-H3-python"))

from export_serve import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
