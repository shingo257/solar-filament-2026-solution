"""v139 cons2of3_sptal_maj3 clean reproduction Colab wrapper contract tests."""

import ast
import hashlib
import json

import pytest

from scratch import build_v139_sptal_clean_repro_colab as builder


def _setup_source():
    nb = json.loads(builder.SOURCE.read_text(encoding="utf-8"))
    return "".join(next(c for c in nb["cells"] if c["cell_type"] == "code")["source"])


def test_patches_select_sptal_trio_and_clean_artifact_root():
    patched = builder.patch_setup(_setup_source())
    assert 'SUBMISSION_VARIANT = "cons2of3_sptal_maj3"' in patched
    assert 'Path("/content/drive/MyDrive/solar_filament_v30")' not in patched
    assert 'DRIVE_ROOT = Path("/content/solar_clean_root")' in patched
    assert 'PREPROC_CACHE = DRIVE_ROOT / "preprocessed_v30_q96"' not in patched
    assert 'DRIVE_ROOT = WORK_ROOT / "solar_filament_v30"' in patched
    assert "24367120279820997396f7063b1ae5b235a442fd1c92617e1e8941d472504c3b" in patched


def test_patch_rejects_missing_anchor():
    with pytest.raises(ValueError, match="anchor not unique"):
        builder.patch_setup(_setup_source().replace('SUBMISSION_VARIANT = "prod"', "X"))


def test_generated_v139_notebook_pins_current_source_and_all_four_weights():
    nb = json.loads(builder.OUTPUT.read_text(encoding="utf-8"))
    sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for src in sources:
        ast.parse(src)
    source_sha = hashlib.sha256(builder.SOURCE.read_bytes()).hexdigest()
    assert f'SOURCE_SHA256 = "{source_sha}"' in sources[0]
    assert builder.REFERENCE_SHA256 in sources[0]
    assert builder.FULL_YOLO_SHA256 in sources[0]
    assert builder.YOLO26_SHA256 in sources[0]
    assert builder.SPTAL_SHA256 in sources[0]
    assert builder.REFINER_ALL_SHA256 in sources[0]
    run_cell = sources[3]
    assign_line = run_cell.splitlines()[1]
    rhs = assign_line.split("SETUP_PATCHES = ", 1)[1]
    assert ast.literal_eval(rhs) == builder.SETUP_PATCHES
