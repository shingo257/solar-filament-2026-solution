"""Contract and behavioral tests for the v114 SAHI tiled-inference notebook (§2.92).

v114 tests sub-1280 upscaled tiled inference (`640²/896² -> imgsz=1280`,
extending beyond §2.21's 1152² 2×2 probe). The tests pin the arm set,
tile geometry, bbox-short-circuited greedy mask-IoU merge equivalence and
runtime budget under DIAG_CONF_FLOOR=0.02, prod-literal defaults (v109
regression guard), and frozen-cell identity.
"""

import ast
import json
import time

import numpy as np
import pytest

from scratch import build_v114_sahi_colab as v114


def _src(cell):
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def _load_wrapper_ns():
    ns = {"np": np, "YOLO": lambda *a, **kw: None}
    exec(compile(v114.V114_WRAPPERS, "<v114_wrappers>", "exec"), ns)  # noqa: S102
    return ns


@pytest.fixture(scope="module")
def nb():
    return json.loads(v114.OUT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def nb_source():
    return json.loads(v114.SOURCE.read_text(encoding="utf-8"))


def test_all_cells_parse(nb):
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        stripped = "\n".join(
            line for line in _src(cell).splitlines() if not line.lstrip().startswith("%"))
        ast.parse(stripped)


def test_config_declares_v114_arms(nb):
    config = _src(nb["cells"][1])
    for arm in ("'sahi640o128'", "'sahi896o192'"):
        assert arm in config, arm
    for key in ("SAHI_TILE = None", "SAHI_OVERLAP = 0",
                "'sahi_tile': 640, 'sahi_overlap': 128",
                "'sahi_tile': 896, 'sahi_overlap': 192"):
        assert key in config, key
    assert "v114_sahi_colab_manifest.json" in config
    assert "GATE_DELTA = 0.010" in config
    for kept in ("PRIMARY_CONF = 0.26", "CROP_SIZE = 320",
                 "REFINER_THRESHOLD = 0.45", "PREDICT_CONF_FLOOR = 0.08"):
        assert kept in config, kept


def test_run_cell_params_and_verdict(nb):
    run = _src(nb["cells"][6])
    assert '"experiment_id": "v114_sahi"' in run
    assert "§2.92" in run
    assert '"params_of_arm"' in run
    assert "reject_v114_fold2" in run
    assert "promote_fold0_replication" in run
    # prod baseline guarded if cell 6 is re-run in same kernel
    assert 'CFG.SAHI_TILE = None; CFG.SAHI_OVERLAP = 0\nif "prod" not in manifest["results"]:' in run
    # per-arm 既定は prod リテラル (v109 汚染の回帰防止)
    assert '"sahi_tile": None, "sahi_overlap": 0' in run
    assert 'CFG.SAHI_TILE = p["sahi_tile"]' in run
    assert 'CFG.SAHI_OVERLAP = p["sahi_overlap"]' in run


def test_sahi_wrapper_geometry_and_merge(nb):
    run = _src(nb["cells"][6])
    # tile coverage: stride = tile-overlap, last tile clamped & anchored to image edge
    assert "stride = tile - overlap" in run
    assert "last = max(length - tile, 0)" in run
    assert "starts.append(last)" in run
    # per-tile predict + survivor-only global canvas reassembly
    assert "cv2.imread(str(source)" in run
    assert "result_masks(result)" in run
    assert "np.zeros((len(kept), height, width), dtype=bool)" in run
    # merge: conf-desc greedy, bbox-short-circuited mask-IoU > 0.30 drop
    assert "iou_thr=0.30" in run
    assert "key=lambda i: -confs[i]" in run
    # prod path preserved when SAHI off
    assert "if CFG.SAHI_TILE is None:" in run
    assert "return _predict(source, **kw)" in run
    # minimal compat namespace same as v109 union pattern
    assert "SimpleNamespace" in run


def test_tile_starts_and_merge_equivalence_and_speed():
    ns = _load_wrapper_ns()
    tile_starts = ns["_tile_starts"]
    tight_crop = ns["_tight_crop"]
    merge_boxes = ns["_merge_sahi_boxes"]
    merge_sahi = ns["_merge_sahi"]

    assert tile_starts(2048, 640, 128) == [0, 512, 1024, 1408]
    assert tile_starts(2048, 896, 192) == [0, 704, 1152]
    assert tile_starts(500, 640, 128) == [0]  # clamped when length < tile

    # Exact bit equivalence against naive full-canvas pairwise boolean IoU
    rng = np.random.default_rng(42)
    h, w = 256, 256
    masks, confs = [], []
    for k in range(25):
        m = np.zeros((h, w), dtype=bool)
        if k > 0:  # include one empty mask at k == 0
            y0, x0 = int(rng.integers(0, 200)), int(rng.integers(0, 200))
            m[y0:y0 + int(rng.integers(8, 45)), x0:x0 + int(rng.integers(8, 45))] = True
        masks.append(m)
        confs.append(float(rng.uniform(0.02, 0.95)))

    order = sorted(range(len(masks)), key=lambda i: -confs[i])
    naive_kept = []
    for i in order:
        dup = False
        for j in naive_kept:
            inter = float((masks[i] & masks[j]).sum())
            union = float((masks[i] | masks[j]).sum())
            if union > 0 and inter / union > 0.30:
                dup = True
                break
        if not dup:
            naive_kept.append(i)

    got_masks, got_confs = merge_sahi(masks, confs, iou_thr=0.30)
    assert len(got_masks) == len(naive_kept)
    assert np.allclose(got_confs, [confs[i] for i in naive_kept])
    for gm, idx in zip(got_masks, naive_kept):
        assert np.array_equal(gm, masks[idx])

    # Performance guard: 200 tile proposals on a 2048x2048 canvas must merge in <0.5s
    crops, boxes, big_confs = [], [], []
    for k in range(200):
        tm = np.zeros((640, 640), dtype=bool)
        ry, rx = int(rng.integers(0, 560)), int(rng.integers(0, 560))
        tm[ry:ry + 40, rx:rx + 40] = True
        ty0 = [0, 512, 1024, 1408][k % 4]
        tx0 = [0, 512, 1024, 1408][(k // 4) % 4]
        b, c = tight_crop(tm, ty0, tx0)
        boxes.append(b)
        crops.append(c)
        big_confs.append(float(rng.uniform(0.02, 0.95)))
    t0 = time.perf_counter()
    kept = merge_boxes(crops, big_confs, boxes, iou_thr=0.30)
    elapsed = time.perf_counter() - t0
    assert len(kept) > 0
    assert elapsed < 0.5, f"_merge_sahi_boxes too slow on 200 proposals: {elapsed:.3f}s"


def test_frozen_cells_unchanged(nb, nb_source):
    for idx in (2, 3, 4, 5):
        assert _src(nb["cells"][idx]) == _src(nb_source["cells"][idx]), f"cell {idx} drifted"


def test_header_declares_fixed_gate_and_caveat(nb):
    hdr = _src(nb["cells"][0])
    assert "§2.92" in hdr
    assert "§2.21" in hdr
    assert "+0.010" in hdr
    assert "sahi640o128" in hdr
    assert "sahi896o192" in hdr
    assert "切断" in hdr or "caveat" in hdr.lower()
