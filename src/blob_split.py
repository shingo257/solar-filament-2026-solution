"""Shim: use ``solar_filament.pipeline.blob_split``."""

from solar_filament.pipeline.blob_split import *  # noqa: F403
from solar_filament.pipeline import blob_split as _mod

__all__ = [name for name in dir(_mod) if not name.startswith("_")]
