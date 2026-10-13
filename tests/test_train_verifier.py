"""train_verifier の前処理契約テスト。"""

import numpy as np

from scratch.train_verifier import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    blend_report,
    to_tensor,
)


def test_to_tensor_keeps_mask_at_unit_scale():
    # mask ch を RGB と一緒に /255 すると 0.0039 に潰れて学習信号を
    # 失う (Codex P1 指摘)。0/1 のままモデルに入れる契約。
    crops = np.zeros((2, 8, 8, 4), dtype=np.uint8)
    crops[0, :, :4, 3] = 1

    tensor = to_tensor(crops)

    assert tensor.shape == (2, 4, 8, 8)
    mask = tensor[:, 3].numpy()
    assert set(np.unique(mask)) <= {0.0, 1.0}
    assert mask[0].sum() == 4 * 8


def test_to_tensor_applies_imagenet_normalization_to_rgb_only():
    crops = np.full((1, 4, 4, 4), 127, dtype=np.uint8)
    crops[..., 3] = 1

    tensor = to_tensor(crops)

    expected = (127 / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    assert abs(float(tensor[0, 0, 0, 0]) - expected) < 1e-4


def test_blend_report_fits_on_select_fold_and_scores_eval_fold():
    # select で係数を学び eval に適用 — verifier が conf と相関するだけの
    # 劣化版なら blend は conf 単独を超えない契約 (RESEARCH_LOG §2.22)。
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 400)
    conf = labels * 0.6 + rng.normal(0, 0.3, 400)          # 強い分離器
    ver = conf * 0.5 + rng.normal(0, 0.4, 400)             # その劣化相関版

    report = blend_report(labels, conf, ver, labels, conf, ver)

    assert 0.5 < report["auc_fold2_blend"] <= 1.0
    assert report["corr_conf_verifier"] > 0.3
