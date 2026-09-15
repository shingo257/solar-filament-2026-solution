"""Shim: use ``solar_filament.pipeline.mask_wbf``."""

from solar_filament.pipeline.mask_wbf import *  # noqa: F403
from solar_filament.pipeline import mask_wbf as _mod

__all__ = [name for name in dir(_mod) if not name.startswith("_")]
