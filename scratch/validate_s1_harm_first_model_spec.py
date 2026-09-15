"""CLI wrapper — library lives in ``solar_filament.s1.harm_first_model_spec``."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solar_filament.s1.harm_first_model_spec import main

if __name__ == "__main__":
    raise SystemExit(main())
