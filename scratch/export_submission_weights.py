"""提出用 weights の repo 同梱エクスポート (2026-10-23 主催者要件対応).

topic 724503: "the weights are stored in the repository and are loaded in
the notebook"。standalone 公開 repo に Git LFS で同梱するための weights
一式を生成し、全 bytes の SHA256 と tensor 等価性を検証する。

エクスポート対象 (submissions prod / plain_mhsa_f2 が必要とする一式):

  source (Drive reviewed artifacts)                  -> repo weights/
  -------------------------------------------------------------------
  runs/yolo_fold_{0,1,2}/weights/best.pt             -> yolo_fold{0,1,2}_best.pt
  artifacts/mediseg_edgeattnet_refiner_all_data_     -> refiner_all_data.pt
      resume.pt (model-only 141MB, そのまま)
  artifacts/ablation_plain_mhsa_seed8407_fold2.pt    -> refiner_plain_mhsa_
      (model+best_model の2重格納 282MB)                 fold2_s8407.pt
                                                       (best_model のみ
                                                        抽出で 141MB)

抽出される `best_model` はローダの `ckpt.get("best_model") or ckpt["model"]`
で原本と同一オブジェクトを返すため、推論は byte ではなく tensor 一致で
等価 (--verify で全 tensor を torch.equal 確認)。

使い方:
  python scratch/export_submission_weights.py --out <dir>
      Drive の原本を読み、--out に weights + manifest を書き出す。
  python scratch/export_submission_weights.py --verify --out <dir>
      --out の生成物が原本と tensor 等価かを検証するのみ。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

DRIVE = Path(os.environ.get("SOLAR_DRIVE_ROOT", "G:/マイドライブ"))
ARTIFACTS = DRIVE / "solar_filament_v30/artifacts"
RUNS = DRIVE / "solar_filament_v30/runs"

# (出力ファイル名, 原本パス, 取り出す state_dict キー or None=全bytes複写)
EXPORTS = [
    ("yolo_fold0_best.pt", RUNS / "yolo_fold_0/weights/best.pt", None),
    ("yolo_fold1_best.pt", RUNS / "yolo_fold_1/weights/best.pt", None),
    ("yolo_fold2_best.pt", RUNS / "yolo_fold_2/weights/best.pt", None),
    ("refiner_all_data.pt",
     ARTIFACTS / "mediseg_edgeattnet_refiner_all_data_resume.pt", None),
    ("refiner_plain_mhsa_fold2_s8407.pt",
     ARTIFACTS / "ablation_plain_mhsa_seed8407_fold2.pt", "best_model"),
]

# frozen grouped_folds.csv (registry copy, notebook が sha256 検証するもの)
GROUPED_FOLDS_SHA256 = (
    "24367120279820997396f7063b1ae5b235a442fd1c92617e1e8941d472504c3b")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_one(out_dir: Path, name: str, source: Path,
               key: str | None) -> dict:
    """原本から出力 file を生成して sha256 等の manifest 行を返す。"""
    if not source.exists():
        raise FileNotFoundError(f"原本なし: {source}")
    target = out_dir / name
    if key is None:
        payload = {"copied": True}
        import shutil
        shutil.copy2(source, target)
        source_sha = sha256_file(source)
    else:
        ckpt = torch.load(source, map_location="cpu", weights_only=False)
        if key not in ckpt:
            raise KeyError(f"{source.name} に {key} キーなし: {list(ckpt)}")
        export = {
            key: ckpt[key],
            "exported_from": source.name,
            "export_key": key,
        }
        torch.save(export, target)
        source_sha = sha256_file(source)
        payload = {"extracted_key": key}
    payload.update({
        "file_name": name,
        "source_path": str(source),
        "source_sha256": source_sha,
        "size_bytes": target.stat().st_size,
        "sha256": sha256_file(target),
    })
    return payload


def verify_one(out_dir: Path, name: str, source: Path, key: str | None) -> str:
    """出力が原本と等価か検証。等価なら "OK"、差分なら詳細を返す。"""
    target = out_dir / name
    if not target.exists():
        return f"MISSING {name}"
    if key is None:
        return (f"OK {name} (byte identical)" if sha256_file(target) ==
                sha256_file(source) else f"BYTE-DIFF {name}")
    src = torch.load(source, map_location="cpu", weights_only=False)
    out = torch.load(target, map_location="cpu", weights_only=False)
    src_sd, out_sd = src[key], out[key]
    if set(src_sd) != set(out_sd):
        return f"KEY-DIFF {name}: {set(src_sd) ^ set(out_sd)}"
    bad = [k for k in src_sd
           if not torch.equal(src_sd[k], out_sd[k])]
    return (f"OK {name} ({len(src_sd)} tensors equal)" if not bad
            else f"TENSOR-DIFF {name}: {bad[:5]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True,
                        help="weights/ 出力先 (LFS 同梱対象ディレクトリ)")
    parser.add_argument("--verify", action="store_true",
                        help="書き出さず原本との等価性のみ検証")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.verify:
        ok = True
        for name, source, key in EXPORTS:
            result = verify_one(args.out, name, source, key)
            print(result)
            ok &= result.startswith("OK")
        return 0 if ok else 1

    rows = []
    for name, source, key in EXPORTS:
        row = export_one(args.out, name, source, key)
        rows.append(row)
        print(f"exported {name}: {row['size_bytes']/1e6:.1f}MB "
              f"sha256={row['sha256'][:16]}...", flush=True)
    manifest = {
        "generated_by": "scratch/export_submission_weights.py",
        "grouped_folds_sha256": GROUPED_FOLDS_SHA256,
        "weights": rows,
    }
    (args.out / "weights_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"manifest -> {args.out/'weights_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
