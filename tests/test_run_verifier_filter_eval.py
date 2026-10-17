"""run_verifier_filter_eval の cache 指紋契約テスト。

verifier スコア・ラベル・refiner・operating point が変わったときに
古い cache を誤用しないことを保証する (Codex P1 指摘への回帰)。
"""

import numpy as np
import pytest

from scratch.run_verifier_filter_eval import (
    config_fingerprint, load_cache, obs_fingerprint)


def _write_inputs(tmp_path, fill=0.5):
    scores = tmp_path / "scores.npy"
    labels = tmp_path / "labels.csv"
    ckpt = tmp_path / "refiner.pt"
    np.save(scores, np.full(4, fill, dtype=np.float32))
    labels.write_text("file_name,prop_idx\na,0\n", encoding="utf-8")
    ckpt.write_bytes(b"weights" + bytes([int(fill * 255)]))
    return scores, labels, ckpt


def test_fingerprint_stable_for_same_inputs(tmp_path):
    scores, labels, ckpt = _write_inputs(tmp_path)
    fp1 = config_fingerprint(scores, labels, ckpt, fold=2)
    fp2 = config_fingerprint(scores, labels, ckpt, fold=2)
    assert fp1 == fp2


def test_fingerprint_changes_when_scores_change(tmp_path):
    scores, labels, ckpt = _write_inputs(tmp_path, fill=0.5)
    fp_base = config_fingerprint(scores, labels, ckpt, fold=2)
    np.save(scores, np.full(4, 0.7, dtype=np.float32))
    assert config_fingerprint(scores, labels, ckpt, fold=2) != fp_base


def test_fingerprint_changes_with_fold_and_refiner(tmp_path):
    scores, labels, ckpt = _write_inputs(tmp_path)
    fp_base = config_fingerprint(scores, labels, ckpt, fold=2)
    assert config_fingerprint(scores, labels, ckpt, fold=1) != fp_base
    ckpt.write_bytes(b"other-weights")
    assert config_fingerprint(scores, labels, ckpt, fold=2) != fp_base


def test_load_cache_rejects_missing_and_mismatched(tmp_path):
    scores, labels, ckpt = _write_inputs(tmp_path)
    fp = config_fingerprint(scores, labels, ckpt, fold=2)
    cache = tmp_path / "obs_fold2.npz"

    assert load_cache(cache, fp) is None  # 非存在

    np.savez_compressed(cache, keep_idx=np.array([0]), fingerprint=fp)
    assert load_cache(cache, fp) is not None          # 指紋一致 → 再利用
    assert load_cache(cache, "deadbeef") is None      # 指紋不一致 → 再計算

    np.savez_compressed(cache, keep_idx=np.array([0]))  # 旧形式 (指紋無し)
    assert load_cache(cache, fp) is None


def test_obs_fingerprint_tracks_proposal_and_image_bytes(tmp_path):
    """OOF npz・前処理画像のいずれかが再生成されると obs 指紋も変わる。"""
    npz = tmp_path / "obs.npz"
    img = tmp_path / "obs.jpg"
    np.savez_compressed(npz, data=np.zeros(4, np.uint8))
    img.write_bytes(b"image-bytes-1")
    base = "globalfp"

    fp1 = obs_fingerprint(base, npz, img)
    assert fp1 == obs_fingerprint(base, npz, img)
    assert fp1 != obs_fingerprint("other-base", npz, img)

    np.savez_compressed(npz, data=np.zeros(5, np.uint8))
    assert obs_fingerprint(base, npz, img) != fp1   # 提案 npz 再生成

    np.savez_compressed(npz, data=np.zeros(4, np.uint8))
    img.write_bytes(b"image-bytes-2")
    assert obs_fingerprint(base, npz, img) != fp1   # 画像再生成
