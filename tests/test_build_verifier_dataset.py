"""build_verifier_dataset のラベル境界とクロップ構築の契約テスト。"""

import numpy as np

from scratch.build_verifier_dataset import CROP, make_crop, positive_label


def test_positive_label_uses_strict_official_iou_boundary():
    # 公式PQは IoU > 0.5。境界値ちょうどは陽性にしない (README.md PQ定義)。
    assert positive_label(0.5) == 0
    assert positive_label(0.4999) == 0
    assert positive_label(0.5001) == 1


def test_make_crop_converts_opencv_bgr_to_rgb():
    # cv2.imdecode は BGR。verifier 入力は ImageNet 前提の RGB でなければ
    # ならないので、0ch に書いた画素は RGB 出力では 2ch に現れる。
    image_bgr = np.zeros((64, 64, 3), dtype=np.uint8)
    image_bgr[:, :, 0] = 200  # BGR の B
    prop = np.zeros((64, 64), dtype=bool)
    prop[8:24, 8:24] = True

    crop = make_crop(image_bgr, prop)

    assert crop.shape == (CROP, CROP, 4)
    assert crop.dtype == np.uint8
    assert crop[..., 0].max() == 0      # RGB の R (= BGR の ch2=0)
    assert crop[..., 2].max() == 200    # RGB の B (= BGR の ch0=200)


def test_make_crop_mask_channel_is_binary():
    image_bgr = np.full((64, 64, 3), 128, dtype=np.uint8)
    prop = np.zeros((64, 64), dtype=bool)
    prop[8:24, 8:24] = True

    mask = make_crop(image_bgr, prop)[..., 3]

    assert set(np.unique(mask)) <= {0, 1}
    assert mask.sum() > 0
