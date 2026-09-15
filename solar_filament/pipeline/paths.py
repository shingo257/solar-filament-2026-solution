"""Shared filesystem roots for competition artifacts.

Drive mounts differ across machines. Prefer ``SOLAR_FILAMENT_DRIVE`` when set,
then common Windows / Colab locations. Callers must treat missing paths as hard
errors rather than silently inventing defaults that look like V30.
"""

from __future__ import annotations

import os
from pathlib import Path


ENV_DRIVE = "SOLAR_FILAMENT_DRIVE"

_DEFAULT_CANDIDATES: tuple[Path, ...] = (
    Path("G:/マイドライブ"),
    Path("/content/drive/MyDrive"),
    Path.home() / "MyDrive",
)


def resolve_drive(explicit: Path | str | None = None) -> Path:
    """Return the Drive (or Drive-equivalent) root that hosts solar_filament_v*.

    Resolution order:
    1. ``explicit`` argument
    2. environment variable ``SOLAR_FILAMENT_DRIVE``
    3. first existing candidate among well-known mount points
    """

    if explicit is not None:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"explicit drive root does not exist: {path}")
        return path

    env = os.environ.get(ENV_DRIVE)
    if env:
        path = Path(env)
        if not path.exists():
            raise FileNotFoundError(f"{ENV_DRIVE}={path} does not exist")
        return path

    for candidate in _DEFAULT_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "no Drive root found; set SOLAR_FILAMENT_DRIVE or pass resolve_drive(path)"
    )


def v30_root(drive: Path | None = None) -> Path:
    return (drive or resolve_drive()) / "solar_filament_v30"


def v34_root(drive: Path | None = None) -> Path:
    return (drive or resolve_drive()) / "solar_filament_v34"


def v35_root(drive: Path | None = None) -> Path:
    return (drive or resolve_drive()) / "solar_filament_v35"


def train_annotations_json(drive: Path | None = None) -> Path:
    return (
        v30_root(drive)
        / "competition_data"
        / "MAGFiLO_1.0_Kaggle_2026"
        / "train"
        / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    )


def grouped_folds_csv(drive: Path | None = None) -> Path:
    return v30_root(drive) / "artifacts" / "grouped_folds.csv"


def oof_cache_root(drive: Path | None = None) -> Path:
    return v34_root(drive) / "prediction_cache" / "oof"


def preprocessed_dir(drive: Path | None = None) -> Path:
    return v30_root(drive) / "preprocessed_v30_q96"


def artifacts_dir(drive: Path | None = None) -> Path:
    return v30_root(drive) / "artifacts"
