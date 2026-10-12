"""v80 候補生成を Colab GPU で実行するための薄いノートブックビルダー。

ロジックは Drive 上のスナップショット
(`<MyDrive>/solar_filament_v80/repo/scratch/run_v80_hybrid_candidates.py`)
をそのまま実行するだけで、コードの複製はしない。
スナップショットはローカル `Solar_Filament_Segmentation/` から
`solar_filament/`・`scratch/`・`src/` をコピーしたもの — コードを変えたら
再コピーすること。

使い方:
    python scratch/build_v80_candidates_colab.py --fold 0
    python scratch/build_v80_candidates_colab.py --fold 1
    python scratch/build_v80_candidates_colab.py --fold 2

GPU必須 (Colab T4/L4)。出力は `<MyDrive>/solar_filament_v80/candidates/fold_{f}/`。
"""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}


def build_cells(fold_num):
    return [
        markdown(
            f"""# V80 — YOLO ∪ component-tree 候補生成: fold{fold_num}

YOLO OOF キャッシュ + V34 semantic 確率マップから候補を作り、fold{fold_num} 専用
refiner (`mediseg_refiner_fold{fold_num}_resume.pt`) を shrink@0.45 で適用して
`<MyDrive>/solar_filament_v80/candidates/fold_{fold_num}/` に保存する。
実行するのは Drive 上の `solar_filament_v80/repo` スナップショットの
`run_v80_hybrid_candidates.py` そのもの(コード複製なし)。

**前提**: ローカル側で `Solar_Filament_Segmentation/{{solar_filament,scratch,src}}` を
`G:\\マイドライブ\\solar_filament_v80\\repo\\` にコピー済みであること。

GPU必須 (T4 で fold あたり 1〜2h 見込み)。
"""
        ),
        code(
            """# [1/3] Drive接続・前提確認
import sys
from pathlib import Path

from google.colab import drive

drive.mount("/content/drive")

REPO = Path("/content/drive/MyDrive/solar_filament_v80/repo")
SCRIPT = REPO / "scratch" / "run_v80_hybrid_candidates.py"
for required in (SCRIPT, REPO / "solar_filament", REPO / "src"):
    if not required.exists():
        raise FileNotFoundError(
            f"{required} がありません。ローカルから Drive へ repo スナップショットを"
            "再コピーしてください (solar_filament/ scratch/ src/ の3ディレクトリ)"
        )
sys.path.insert(0, str(REPO))
!pip -q install pycocotools
import torch
assert torch.cuda.is_available(), "GPUランタイムを選択してください"
print({"GPU": torch.cuda.get_device_name(0), "repo": str(REPO)})
"""
        ),
        code(
            f"""# [2/3] fold{fold_num} の候補生成 (GPU)
import importlib.util

spec = importlib.util.spec_from_file_location("v80_candidates", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)  # __name__ != "__main__" なので main() は走らない

sys.argv = [
    "run_v80_hybrid_candidates.py",
    "--fold", "{fold_num}",
    "--threads", "2",           # CPU 部分 (cv2/tree) のみ。refiner は GPU
    "--skip-existing",          # 中断再開に対応
]
mod.main()
"""
        ),
        code(
            f"""# [3/3] manifest 確認
import json

manifest = json.loads(
    (REPO.parent / "candidates" / "manifest_fold{fold_num}.json")
    .read_text(encoding="utf-8"))
manifest.pop("input_fingerprints", None)
print(json.dumps(manifest, ensure_ascii=False, indent=2))
"""
        ),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    notebook = {
        "cells": build_cells(args.fold),
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = args.out or ROOT / f"solar_filament_v80_candidates_fold{args.fold}_colab.ipynb"
    out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
