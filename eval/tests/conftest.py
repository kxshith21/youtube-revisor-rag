"""Test bootstrap: make eval modules importable without installing a package."""

from __future__ import annotations

import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[1]
ROOT = EVAL_DIR.parent

for path in (str(EVAL_DIR), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)
