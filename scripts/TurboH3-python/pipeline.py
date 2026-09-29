#!/usr/bin/env python3
"""TurboH3 pipeline runner — a thin wrapper over MiniMax-H3-python.

The LightX2V Turbo artifacts are few-step LoRA students of MiniMax-H3:
same base checkpoint, same encoders/VAEs, loaded through the dynamic
LoRA route (--lora-backend peft --lora-path <artifact>) instead of
load-time fusion. Deployment and request generation therefore reuse the
MiniMax-H3-python machinery; only the knobs differ (see configs/ and
README.md): the request must carry the artifact's own contract — its
step count and flow shift from the recipe table — plus the request-level
"lora" activation field.

Usage (from the repo root):
    python scripts/TurboH3-python/pipeline.py \
        --config scripts/TurboH3-python/configs/ref2va/rtx5090/config.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "MiniMax-H3-python"))

from pipeline import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
