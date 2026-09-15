"""Shared pytest fixtures and import path for Solar Filament Segmentation.

pyproject.toml already sets ``pythonpath = ["."]``. This conftest keeps the
project root discoverable when tests are invoked as plain scripts or from a
parent directory without installing the package.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)
