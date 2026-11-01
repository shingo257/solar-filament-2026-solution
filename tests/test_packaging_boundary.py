"""Regression test for the wheel package boundary (pyproject.toml).

PR #596 removed ``src*``/``scratch*`` from ``packages.find`` so the installed
wheel carries only the canonical ``solar_filament`` package. CI runs pytest
against the source tree (``--no-project`` + pythonpath), which cannot observe
the wheel contents — this test resolves the actual ``packages.find`` spec so
a regression in the include list is caught locally.

``setuptools.find_packages`` cannot be used directly: the CI environment is
``uv run --no-project`` with an explicit ``--with`` list that does not ship
setuptools. The include spec is a simple fnmatch list over dotted package
names, so the equivalent resolution is a directory scan for ``__init__.py``.
"""

from __future__ import annotations

import tomllib
from fnmatch import fnmatch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"


def _include_spec() -> list[str]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return data["tool"]["setuptools"]["packages"]["find"]["include"]


def _resolved_packages() -> set[str]:
    """Equivalent of setuptools.find_packages for flat include globs."""

    found: set[str] = set()
    for init in ROOT.rglob("__init__.py"):
        name = ".".join(init.parent.relative_to(ROOT).parts)
        if any(fnmatch(name, pattern) for pattern in _include_spec()):
            found.add(name)
    return found


def test_include_spec_is_solar_filament_only():
    assert _include_spec() == ["solar_filament*"]


def test_resolved_packages_carry_solar_filament():
    packages = _resolved_packages()
    assert "solar_filament" in packages
    assert "solar_filament.metrics" in packages
    assert "solar_filament.pipeline" in packages


def test_resolved_packages_exclude_src_and_scratch():
    packages = _resolved_packages()
    assert not any(p == "src" or p.startswith("src.") for p in packages)
    assert not any(p == "scratch" or p.startswith("scratch.") for p in packages)
