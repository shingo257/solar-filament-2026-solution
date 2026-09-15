"""提出CSVの検証器と RLE codec のテスト。

HANDOVER §29 が指摘する通り、validator・metric・mask cluster には体系的な自動テストが
無かった。V18〜V25 は提出形式エラーだけで何度も弾かれており、ここは実際に壊れた箇所である。

特に重要なのが **純Python codec と pycocotools の一致検証**。
V20 で「Internet なし・pycocotools なしでも compressed RLE を書ける」ようにしたが、
その出力が公式の pycocotools と本当に同じかは一度も確かめられていなかった。
食い違えば提出は黙って壊れる。

pytest でも `python tests/test_submission.py` でも走る。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from solar_filament.pipeline.v32_overlap_cluster import (  # noqa: E402
    HEIGHT,
    WIDTH,
    decode_mask,
    encode_mask,
    load_submission,
    validate_submission,
)

try:
    from pycocotools import mask as coco_mask
    HAS_PYCOCOTOOLS = True
except ImportError:  # 公式RLEとの照合はスキップするが、他のテストは走らせる
    HAS_PYCOCOTOOLS = False


def _blank():
    return np.zeros((HEIGHT, WIDTH), dtype=bool)


def _rect(y0, y1, x0, x1):
    mask = _blank()
    mask[y0:y1, x0:x1] = True
    return mask


def _frame(rows):
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


def _expect_failure(callable_, needle):
    try:
        callable_()
    except AssertionError as error:
        assert needle.lower() in str(error).lower(), f"想定外のメッセージ: {error}"
        return
    raise AssertionError(f"検出されるべき異常が通ってしまった: {needle}")


# ---------------------------------------------------------------------------
# RLE codec
# ---------------------------------------------------------------------------


def test_codec_roundtrip_random():
    rng = np.random.default_rng(3)
    for _ in range(5):
        mask = _blank()
        for _ in range(4):
            y, x = rng.integers(0, 1900, 2)
            h, w = rng.integers(10, 140, 2)
            mask[y:y + h, x:x + w] = True
        assert np.array_equal(decode_mask(encode_mask(mask)), mask)


def test_codec_handles_thin_and_edge_masks():
    """細長い個体と画像端は RLE が壊れやすい典型。"""
    for mask in (_rect(1000, 1001, 0, 2048),      # 1px 高さの横一線
                 _rect(0, 2048, 1024, 1025),      # 1px 幅の縦一線
                 _rect(0, 3, 0, 3),               # 左上隅
                 _rect(2045, 2048, 2045, 2048)):  # 右下隅
        assert np.array_equal(decode_mask(encode_mask(mask)), mask)


def test_pure_python_codec_matches_pycocotools():
    """★ 純Python codec が公式 pycocotools と厳密に一致すること。

    ここが食い違うと、オフライン生成した提出が黙って壊れる。
    """
    if not HAS_PYCOCOTOOLS:
        print("  SKIP: pycocotools 未導入")
        return
    rng = np.random.default_rng(11)
    for _ in range(6):
        mask = _blank()
        for _ in range(5):
            y, x = rng.integers(0, 1900, 2)
            h, w = rng.integers(5, 200, 2)
            mask[y:y + h, x:x + w] = True

        ours = encode_mask(mask)
        official = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("ascii")
        assert ours == official, "encode が公式と不一致"

        # 公式が書いた文字列を自前 decoder が読めること（逆方向）
        assert np.array_equal(decode_mask(official), mask), "decode が公式出力を読めない"


# ---------------------------------------------------------------------------
# 提出CSVの検証器 — HANDOVER §4.4 のチェックリストに対応
# ---------------------------------------------------------------------------


def test_validator_accepts_clean_submission():
    frame = _frame([
        ("img_1", encode_mask(_rect(10, 40, 10, 40))),
        ("img_2", encode_mask(_rect(100, 140, 100, 140))),
        ("other_1", encode_mask(_rect(10, 40, 10, 40))),   # 別画像なら同座標でもよい
    ])
    report = validate_submission(frame)
    assert report["rows"] == 3
    assert report["images"] == 2
    assert report["overlap_pixels"] == 0
    assert report["area_min"] > 0


def test_validator_rejects_wrong_columns():
    """旧案の `rle_mask` は誤り。列名は正確に filament_id / segmentation_rle。"""
    frame = pd.DataFrame([("img_1", encode_mask(_rect(10, 40, 10, 40)))],
                         columns=["filament_id", "rle_mask"])
    _expect_failure(lambda: validate_submission(frame), "rle_mask")


def test_validator_rejects_column_order_swap():
    frame = pd.DataFrame([(encode_mask(_rect(10, 40, 10, 40)), "img_1")],
                         columns=["segmentation_rle", "filament_id"])
    _expect_failure(lambda: validate_submission(frame), "segmentation_rle")


def test_validator_rejects_nulls():
    """Kaggle の `Submission contains null values` を再現する。"""
    frame = _frame([("img_1", encode_mask(_rect(10, 40, 10, 40))), ("img_2", None)])
    _expect_failure(lambda: validate_submission(frame), "null")


def test_validator_rejects_empty_frame():
    _expect_failure(lambda: validate_submission(_frame([])), "empty")


def test_validator_rejects_duplicate_filament_id():
    rle = encode_mask(_rect(10, 40, 10, 40))
    frame = _frame([("img_1", rle), ("img_1", rle)])
    _expect_failure(lambda: validate_submission(frame), "duplicate")


def test_validator_rejects_empty_mask():
    """空maskは行ごと出力してはいけない（空文字やNaNで残さない）。"""
    frame = _frame([("img_1", encode_mask(_rect(10, 40, 10, 40))),
                    ("img_2", encode_mask(_blank()))])
    _expect_failure(lambda: validate_submission(frame), "empty mask")


def test_validator_rejects_overlapping_masks():
    """Kaggle の `Submissions may not contain overlapping masks` を再現する。

    1画素でも重なれば拒否されること。
    """
    frame = _frame([("img_1", encode_mask(_rect(10, 40, 10, 40))),
                    ("img_2", encode_mask(_rect(39, 60, 39, 60)))])   # 1画素だけ重なる
    _expect_failure(lambda: validate_submission(frame), "overlap")


def test_validator_allows_touching_but_disjoint_masks():
    """接しているだけ（共有画素なし）は正常。過検出しないこと。"""
    frame = _frame([("img_1", encode_mask(_rect(10, 40, 10, 40))),
                    ("img_2", encode_mask(_rect(40, 60, 10, 40)))])
    assert validate_submission(frame)["overlap_pixels"] == 0


def test_image_id_split_keeps_string_ids():
    """COCO の image id は文字列。`int()` へ変換してはいけない（§4.1）。

    validator は filament_id の末尾 `_n` だけを落として画像IDを作る。
    `040301-20140609195854Bh` のようなIDが壊れないこと。
    """
    stem = "040301-20140609195854Bh"
    frame = _frame([(f"{stem}_1", encode_mask(_rect(10, 40, 10, 40))),
                    (f"{stem}_2", encode_mask(_rect(100, 140, 100, 140)))])
    report = validate_submission(frame)
    assert report["images"] == 1, "同一画像の2個体が別画像として数えられている"


# ---------------------------------------------------------------------------
# 実際の提出候補に対する結合テスト
# ---------------------------------------------------------------------------


def test_real_v32_submission_passes_validator():
    """Kaggle が受理した実物 (V32b, Public 0.35) が今も検証を通ること。

    全1406行を2048^2で復号すると重いので先頭のみ。形式の回帰検出が目的。
    """
    path = ROOT / "output" / "v32" / "submission_v32b_containment099_dice050.csv"
    if not path.exists():
        print(f"  SKIP: {path.name} がない")
        return
    frame = load_submission(path)
    assert frame.columns.tolist() == ["filament_id", "segmentation_rle"]
    assert not frame.isna().any().any(), "実提出CSVにnullがある"
    assert not frame["filament_id"].duplicated().any(), "実提出CSVにID重複がある"

    subset = frame.head(30).reset_index(drop=True)
    report = validate_submission(subset)
    assert report["overlap_pixels"] == 0
    assert report["area_min"] > 0
    print(f"  実提出CSV {len(frame)}行のうち先頭{len(subset)}行を検証: "
          f"面積 min={report['area_min']} median={report['area_median']} max={report['area_max']}")


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print(f"PASS {name}")
            except AssertionError as error:
                failures += 1
                print(f"FAIL {name}: {error}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
