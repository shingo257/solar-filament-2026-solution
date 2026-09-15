"""Shim: use ``solar_filament.pipeline.paths``."""

from solar_filament.pipeline.paths import *  # noqa: F403
from solar_filament.pipeline import paths as _mod

__all__ = [name for name in dir(_mod) if not name.startswith("_")]
