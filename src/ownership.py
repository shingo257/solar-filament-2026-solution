"""Shim: use ``solar_filament.pipeline.ownership``."""

from solar_filament.pipeline.ownership import *  # noqa: F403
from solar_filament.pipeline import ownership as _mod

__all__ = [name for name in dir(_mod) if not name.startswith("_")]
