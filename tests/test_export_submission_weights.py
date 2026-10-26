"""export_submission_weights の抽出・検証ロジックの契約テスト。

実際の Drive 重みは不要。合成 checkpoint で以下を確認する:
- key=None は byte 一致コピー
- key 指定は該当 state_dict のみ抽出し metadata を付与
- --verify が tensor 等価を受理し、差分/欠損を拒否する
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

MODULE = Path(__file__).resolve().parents[1]
SCRIPT = MODULE / "scratch" / "export_submission_weights.py"
spec = importlib.util.spec_from_file_location("export_submission_weights",
                                              SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def source_ckpt(tmp_path):
    """model + best_model の2重格納を持つ原本を作る。"""
    sd = {f"w{i}": torch.full((4, 4), float(i)) for i in range(3)}
    ckpt = {"model": sd, "best_model": sd, "epochs": 5, "lr": 1e-4}
    src = tmp_path / "src.pt"
    torch.save(ckpt, src)
    return src, sd


def test_copy_export_is_byte_identical(tmp_path):
    src = tmp_path / "plain.pt"
    src.write_bytes(b"\x01\x02" * 500_000)
    row = mod.export_one(tmp_path, "copy.pt", src, None)
    assert row["sha256"] == _sha(src)
    assert (tmp_path / "copy.pt").read_bytes() == src.read_bytes()


def test_key_export_extracts_only_that_key(tmp_path, source_ckpt):
    src, sd = source_ckpt
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    row = mod.export_one(out_dir, "sub.pt", src, "best_model")
    saved = torch.load(out_dir / "sub.pt", map_location="cpu",
                       weights_only=False)
    assert set(saved["best_model"]) == set(sd)
    assert "model" not in saved  # 2重格納を剥がす
    assert saved["export_key"] == "best_model"
    assert row["sha256"] == _sha(out_dir / "sub.pt")
    assert row["size_bytes"] <= src.stat().st_size


def test_key_export_missing_key_raises(tmp_path):
    src = tmp_path / "empty.pt"
    torch.save({"model": {}}, src)
    (tmp_path).joinpath("o").mkdir(exist_ok=True)
    with pytest.raises(KeyError):
        mod.export_one(tmp_path / "o", "x.pt", src, "best_model")


def test_verify_passes_on_equal_tensors(tmp_path, source_ckpt):
    src, _ = source_ckpt
    out_dir = tmp_path / "w"
    out_dir.mkdir()
    mod.export_one(out_dir, "sub.pt", src, "best_model")
    assert mod.verify_one(out_dir, "sub.pt", src, "best_model").startswith("OK")


def test_verify_rejects_tensor_diff(tmp_path, source_ckpt):
    src, sd = source_ckpt
    out_dir = tmp_path / "w"
    out_dir.mkdir()
    mod.export_one(out_dir, "sub.pt", src, "best_model")
    # 1 tensor だけ書き換えた偽物を置く (w0 は元々全0なので w1 を改ざん)
    saved = torch.load(out_dir / "sub.pt", map_location="cpu",
                       weights_only=False)
    saved["best_model"]["w1"] = torch.zeros_like(saved["best_model"]["w1"])
    torch.save(saved, out_dir / "sub.pt")
    assert mod.verify_one(out_dir, "sub.pt", src,
                          "best_model").startswith("TENSOR-DIFF")


def test_verify_rejects_missing_file(tmp_path, source_ckpt):
    src, _ = source_ckpt
    assert mod.verify_one(tmp_path, "absent.pt", src,
                          "best_model").startswith("MISSING")
