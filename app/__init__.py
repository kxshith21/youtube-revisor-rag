"""YouTube Reviser API package.

The pipeline modules live in ``scripts/`` and are imported as top-level
modules (they already expect their own directory on ``sys.path``). Add that
directory once here so ``app`` modules can simply ``from retrieve import ...``
without duplicating path setup.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
