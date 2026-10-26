"""v73 SSL refiner パイプラインの契約テスト。

- encoder SSL ckpt は inc/down1-4 のみを含み、MediSegUNet / EdgeAttUNet の
  両方へ strict=False でロードできる (decoder/edge_attn は missing)
- apply_encoder_init は encoder 外キーを持つ ckpt を拒否する
- solar_three_channel / make_view は下流 crop と同じ 3ch・320px 契約
- InstanceCropDataset は v35 と同じ record 単位・aug 契約
"""

import numpy as np
import pytest
import torch

from solar_filament.models.mediseg_models import (
    MediSegEdgeAttUNet, MediSegUNet)
from scratch.pretrain_unet_encoder_ssl import (
    encoder_feature, encoder_state_dict, make_view, nt_xent,
    solar_three_channel)
from scratch.train_refiner_fold import (
    InstanceCropDataset, apply_encoder_init, square_box_from_bbox)

def _save_encoder(state, path):
    torch.save({"encoder": state, "corpus_provenance": {
        "guard_policy": "train-test-temporal-v2", "config_fingerprint": "a" * 16,
        "corpus_config_sha256": "b" * 64, "manifest_sha256": "c" * 64,
        "frames": 2, "exclusion_minutes": 360,
        "train_observations_sha256": "d" * 64,
        "test_observations_sha256": "e" * 64}}, path)


def _encoder_only_state(seed=0):
    torch.manual_seed(seed)
    model = MediSegUNet(3, 1, False)
    return {k: v.clone() for k, v in model.state_dict().items()
            if k.startswith(("inc.", "down"))}


class TestEncoderInit:
    def test_ssl_state_loads_into_unet_and_edgeatt(self, tmp_path):
        ckpt = tmp_path / "enc.pt"
        _save_encoder(_encoder_only_state(), ckpt)
        for cls in (MediSegUNet, MediSegEdgeAttUNet):
            model = cls(3, 1, False)
            report = apply_encoder_init(model, ckpt)
            assert len(report["loaded"]) == 60
            missing = report["missing"]
            assert missing, "decoder keys must remain random-init"
            assert all(not m.startswith(("inc.", "down")) for m in missing)

    def test_rejects_non_encoder_keys(self, tmp_path):
        ckpt = tmp_path / "bad.pt"
        state = _encoder_only_state()
        state["up1.conv.double_conv.0.weight"] = torch.zeros(256, 512, 3, 3)
        _save_encoder(state, ckpt)
        with pytest.raises(ValueError, match="encoder 以外"):
            apply_encoder_init(MediSegUNet(3, 1, False), ckpt)

    def test_init_actually_copies_weights(self, tmp_path):
        ckpt = tmp_path / "enc.pt"
        _save_encoder(_encoder_only_state(seed=7), ckpt)
        model = MediSegUNet(3, 1, False)
        before = model.inc.double_conv[0].weight.clone()
        apply_encoder_init(model, ckpt)
        assert not torch.equal(before, model.inc.double_conv[0].weight)

    def test_missing_state_dict_raises(self, tmp_path):
        ckpt = tmp_path / "empty.pt"
        torch.save({}, ckpt)
        with pytest.raises(Exception):
            apply_encoder_init(MediSegUNet(3, 1, False), ckpt)


class TestEncoderStateDict:
    def test_roundtrip_keys(self):
        sd = encoder_state_dict(MediSegEdgeAttUNet(3, 1, False))
        assert sd and all(k.startswith(("inc.", "down")) for k in sd)
        model = MediSegUNet(3, 1, False)
        res = model.load_state_dict(sd, strict=False)
        assert not res.unexpected_keys

    def test_feature_dim(self):
        model = MediSegUNet(3, 1, False)
        feat = encoder_feature(model, torch.zeros(2, 3, 64, 64))
        assert feat.shape == (2, 1024)


class TestSolarChannelAndViews:
    def test_solar3_contract(self):
        gray = np.zeros((512, 512), np.uint8)
        gray[64:448, 64:448] = 180  # disk
        out = solar_three_channel(gray)
        assert out.shape == (512, 512, 3) and out.dtype == np.uint8
        assert out[10, 10].max() == 0  # disk 外は 0 マスク

    def test_make_view_shape_and_independence(self):
        gray = np.zeros((512, 512), np.uint8)
        gray[64:448, 64:448] = 180
        frames3 = np.stack([solar_three_channel(gray)] * 4)
        rng = np.random.default_rng(0)
        v = make_view(rng, frames3, 320)
        assert v.shape == (4, 3, 320, 320)
        assert v.dtype == np.float32 and 0.0 <= v.min() and v.max() <= 1.0
        v2 = make_view(np.random.default_rng(1), frames3, 320)
        assert not np.array_equal(v, v2)

    def test_nt_xent_perfect_positive_low(self):
        z = torch.randn(8, 128)
        assert nt_xent(z, z.clone()).item() < nt_xent(
            z, torch.randn(8, 128)).item()


class TestCropDataset:
    def _toy_coco(self, tmp_path):
        from pycocotools import mask as coco_mask
        image = np.zeros((2048, 2048), np.uint8)
        image[500:700, 500:900] = 200
        cv2 = pytest.importorskip("cv2")
        name = "test.jpeg"
        cv2.imwrite(str(tmp_path / name), np.dstack([image] * 3))
        rle = coco_mask.encode(np.asfortranarray(
            np.pad(np.ones((100, 200)), ((500, 1448), (500, 1348))
                   ).astype(np.uint8)))
        images = {1: {"id": 1, "file_name": name,
                      "height": 2048, "width": 2048}}
        annotations = {1: [{"image_id": 1, "bbox": [500, 500, 200, 100],
                            "area": 20000, "segmentation": rle,
                            "iscrowd": 0}]}
        file_to_ids = {name: [1]}
        return images, annotations, file_to_ids, coco_mask, name

    def test_dataset_item_contract(self, tmp_path):
        images, annotations, file_to_ids, coco_mask, name = self._toy_coco(tmp_path)
        ds = InstanceCropDataset([name], file_to_ids, annotations, images,
                                 tmp_path, coco_mask, augment=False)
        assert len(ds) == 1
        image, target = ds[0]
        assert image.shape == (3, 320, 320) and target.shape == (1, 320, 320)
        assert target.max() == 1.0  # GT が crop 内に存在

    def test_square_box_bounds(self):
        x0, y0, x1, y1 = square_box_from_bbox([0, 0, 100, 50], 2048, 2048)
        assert x0 >= 0 and y0 >= 0 and x1 <= 2048 and y1 <= 2048
        assert x1 - x0 == y1 - y0  # square
