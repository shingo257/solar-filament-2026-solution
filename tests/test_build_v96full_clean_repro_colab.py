"""v96-full clean reproduction Colab wrapper: patches and pinned inputs stay consistent."""

import ast
import hashlib
import json
from pathlib import Path

import pytest

from scratch import build_v96full_clean_repro_colab as builder

MODULE = Path(__file__).resolve().parents[1]


def _setup_source():
    nb = json.loads(builder.SOURCE.read_text(encoding="utf-8"))
    return "".join(next(c for c in nb["cells"] if c["cell_type"] == "code")["source"])


def test_patches_select_v96full_and_a_clean_artifact_root():
    patched = builder.patch_setup(_setup_source())
    assert 'SUBMISSION_VARIANT = "prod_v96full"' in patched
    assert 'Path("/content/drive/MyDrive/solar_filament_v30")' not in patched
    assert 'DRIVE_ROOT = Path("/content/solar_clean_root")' in patched
    assert 'PREPROC_CACHE = DRIVE_ROOT / "preprocessed_v30_q96"' not in patched
    # Kaggle branch and the frozen-fold pin are left untouched
    assert 'DRIVE_ROOT = WORK_ROOT / "solar_filament_v30"' in patched
    assert "24367120279820997396f7063b1ae5b235a442fd1c92617e1e8941d472504c3b" in patched


def test_patch_rejects_missing_anchor():
    with pytest.raises(ValueError, match="anchor not unique"):
        builder.patch_setup(_setup_source().replace('SUBMISSION_VARIANT = "prod"', "X"))


def test_generated_notebook_pins_current_source_and_parses():
    nb = json.loads(builder.OUTPUT.read_text(encoding="utf-8"))
    sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for src in sources:
        ast.parse(src)
    source_sha = hashlib.sha256(builder.SOURCE.read_bytes()).hexdigest()
    assert f'SOURCE_SHA256 = "{source_sha}"' in sources[0]
    assert builder.REFERENCE_SHA256 in sources[0]
    assert builder.FULL_YOLO_SHA256 in sources[0]
    run_cell = sources[3]
    namespace = {}
    exec(run_cell.split("\n", 2)[1], namespace)  # SETUP_PATCHES literal
    assert namespace["SETUP_PATCHES"] == builder.SETUP_PATCHES
