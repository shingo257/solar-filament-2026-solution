"""End-to-end smoke: spine pseudo-instances through the real YOLO transform chain.

Builds a tiny synthetic seg dataset (cls=0 polygons + cls=1 spine rows,
nc=2), instantiates YOLODataset with augmentation on, inserts
SplitSpineSegments before Format, and verifies the pseudo-instances ride
the geometric transforms and are stripped before mask rasterization.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ultralytics")

from solar_filament.pipeline.spine_yolo import SPINE_CLS, SplitSpineSegments

IMG = 640


def _make_dataset_dir(tmp_path, n_images=6):
    import cv2
    img_dir = tmp_path / "images" / "train"
    lb_dir = tmp_path / "labels" / "train"
    img_dir.mkdir(parents=True)
    lb_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n_images):
        img = rng.integers(80, 160, (IMG, IMG, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"im{i}.png"), img)
        # one real polygon + one spine pseudo-row (normalized)
        poly = "0 " + " ".join(
            f"{v:.4f}" for v in [0.2, 0.2, 0.6, 0.2, 0.6, 0.6, 0.2, 0.6])
        spine = f"{SPINE_CLS} " + " ".join(
            f"{v:.4f}" for v in [0.25, 0.4, 0.5, 0.45, 0.75, 0.4])
        (lb_dir / f"im{i}.txt").write_text(poly + "\n" + spine + "\n")
    return img_dir


def _hyp():
    from ultralytics.utils import IterableSimpleNamespace
    from ultralytics.cfg import DEFAULT_CFG_DICT
    hyp = dict(DEFAULT_CFG_DICT)
    hyp.update({"mosaic": 1.0, "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0,
                "degrees": 5.0, "translate": 0.1, "scale": 0.3, "shear": 0.0,
                "perspective": 0.0, "flipud": 0.0, "fliplr": 0.5,
                "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0, "erasing": 0.0})
    return IterableSimpleNamespace(**hyp)


def _dataset(tmp_path):
    from ultralytics.data.dataset import YOLODataset
    img_dir = _make_dataset_dir(tmp_path)
    data = {"nc": 2, "names": {0: "filament", 1: "spine_aux"},
            "channels": 3, "path": str(tmp_path)}
    ds = YOLODataset(img_path=str(img_dir), imgsz=IMG, batch_size=4,
                     augment=True, hyp=_hyp(), rect=False, cache=False,
                     single_cls=False, pad=0.0, prefix="smoke", task="segment",
                     data=data, fraction=1.0)
    return ds


def test_pseudo_instances_survive_pipeline(tmp_path):
    np.random.seed(0)
    ds = _dataset(tmp_path)
    # insert split just before the Format transform (last in Compose)
    from ultralytics.data.augment import Format
    transforms = ds.transforms.transforms
    fmt_idx = next(i for i, t in enumerate(transforms) if isinstance(t, Format))
    transforms.insert(fmt_idx, SplitSpineSegments())
    got = 0
    for _ in range(10):
        labels = ds[np.random.randint(len(ds))]
        segs = labels["spine_segments_px"]
        got += len(segs)
        # every remaining instance is a real class-0 filament
        cls = np.asarray(labels["cls"]).reshape(-1)
        assert (cls == 0).all(), f"pseudo-class leaked: {cls}"
        for s in segs:
            s = np.asarray(s)
            assert s.ndim == 2 and s.shape[1] == 2
    assert got > 0, "no spine pseudo-instances reached the split"


def test_val_pipeline_has_no_spine(tmp_path):
    ds = _dataset(tmp_path)
    ds.transforms = ds.build_transforms(hyp=_hyp())
    ds.augment = False
    from ultralytics.data.augment import Compose, LetterBox, Format
    ds.transforms = Compose([LetterBox(new_shape=(IMG, IMG), scaleup=False),
                             SplitSpineSegments(),
                             Format(bbox_format="xywh", normalize=True,
                                    return_mask=True, batch_idx=True,
                                    mask_ratio=4, mask_overlap=True)])
    labels = ds[0]
    # val labels still contain the cls=1 row (no spine files emitted for val
    # in production; the split transform must also strip any that appear)
    cls = np.asarray(labels["cls"]).reshape(-1)
    assert (cls == 0).all()
