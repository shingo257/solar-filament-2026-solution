"""Drive persistence must fail closed before claiming a durable mirror."""
import json
import os
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("mode", ["mount_error", "not_mounted", "corrupt_copy", "valid"])
def test_mirror_checks_mount_and_destination_bytes(tmp_path, monkeypatch, mode):
    work = tmp_path / "work"
    work.mkdir()
    (work / "result.json").write_text('{"ok":true}', encoding="utf-8")
    drive_root = tmp_path / "drive"
    (drive_root / "MyDrive").mkdir(parents=True)

    def mount(path):
        if mode == "mount_error":
            raise RuntimeError("mount failed")

    google = ModuleType("google")
    colab = ModuleType("google.colab")
    colab.drive = SimpleNamespace(mount=mount)
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setattr(os.path, "ismount", lambda path: mode != "not_mounted")
    original_copy = shutil.copy2

    def copy(source, destination):
        original_copy(source, destination)
        if mode == "corrupt_copy":
            Path(destination).write_bytes(b"corrupt")

    monkeypatch.setattr(shutil, "copy2", copy)
    notebook = json.loads((ROOT / "solar_filament_v73_ssl_refiner_colab.ipynb").read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][-1]["source"])
    scope = {"Path": lambda p: (drive_root / str(p)[len("/content/drive"):].lstrip("/"))
             if str(p).startswith("/content/drive") else Path(p),
             "WORK": work, "json": json}
    if mode == "valid":
        exec(compile(source, "mirror-cell", "exec"), scope)  # noqa: S102 - repository cell with mocked Drive
        manifests = list(drive_root.rglob("sha256_manifest.json"))
        assert len(manifests) == 1
        assert "result.json" in json.loads(manifests[0].read_text())
    else:
        with pytest.raises(RuntimeError):
            exec(compile(source, "mirror-cell", "exec"), scope)  # noqa: S102 - repository cell with mocked Drive
        assert not list(drive_root.rglob("sha256_manifest.json"))
        if mode in ("mount_error", "not_mounted"):
            assert list((drive_root / "MyDrive").iterdir()) == []
    assert (work / "result.json").read_bytes() == b'{"ok":true}'
