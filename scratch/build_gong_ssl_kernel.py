"""GONG SSL verifier 実験の Kaggle ノートブック (solar_filament_v72_gong_ssl) を生成する。

v70 の残存仮説: verifier が conf に届かないのは「2.6k crops で画像特徴が
学べない」ためか。GONG 無ラベルコーパス (1,889 全日面フレーム、test ±6h
除外済み) で ResNet18 を SimCLR 事前学習し、verifier の ImageNet init と
差し替えて fold2 AUC を比較する。

棄却ゲート (事前固定): ssl init の fold2 AUC が imagenet init を有意に
上回らない、または blend が conf 単独 (0.797) を超えない → 棄却。

データセット (private):
  - shingo257/solar-gong-ssl      : corpus/*.npz + code/*.py
  - shingo257/solar-verifier-crops: verifier/crops_fold*.npy + labels_fold*.csv
"""

from __future__ import annotations

import json
from pathlib import Path

CELLS = [
    {
        "cell_type": "markdown",
        "source": [
            "# v72 — GONG SimCLR 事前学習 → verifier init 交換実験\n",
            "\n",
            "v70 (RESEARCH_LOG §2.22) で深層 verifier は fold2 AUC 0.638 vs conf 0.797 で棄却。\n",
            "唯一残った仮説「小データで視覚特徴が学べない」を、無ラベル GONG 全日面コーパスでの\n",
            "ドメイン事前学習で検証する。test 観測 ±6h のフレームは取得時に除外済み\n",
            "(manifest に URL/sha256 記録)。\n",
            "\n",
            "**棄却ゲート**: ssl init の fold2 AUC が imagenet init を上回らなければ棄却。\n",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "import os, shutil, subprocess, sys, json\n",
            "from pathlib import Path\n",
            "\n",
            "IN = Path('/kaggle/input')\n",
            "print(sorted(str(p.relative_to(IN)) for p in IN.rglob('*') if p.is_dir()))\n",
            "# マウントレイアウト差に備えて glob で解決 (/kaggle/input/datasets/<user>/<slug> 型を含む)\n",
            "corpus_dir = next(p.parent for p in IN.rglob('batch_*.npz'))\n",
            "ver_dir = next(p.parent for p in IN.rglob('crops_fold0.npy'))\n",
            "code_dir = next(p.parent for p in IN.rglob('pretrain_gong_ssl.py'))\n",
            "WORK = Path('/kaggle/working')\n",
            "print('corpus:', corpus_dir, sorted(p.name for p in corpus_dir.iterdir())[:4])\n",
            "print('verifier:', ver_dir, sorted(p.name for p in ver_dir.iterdir()))\n",
            "print('code:', code_dir, sorted(p.name for p in code_dir.iterdir()))\n",
            "print('cuda:', __import__('torch').cuda.is_available(),\n",
            "      __import__('torch').cuda.get_device_name(0) if __import__('torch').cuda.is_available() else '')\n",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "%%time\n",
            "# SimCLR 事前学習 (ResNet18, 128px crop, NT-Xent)\n",
            "r = subprocess.run([sys.executable, str(code_dir / 'pretrain_gong_ssl.py'),\n",
            "                    '--corpus', str(corpus_dir),\n",
            "                    '--epochs', '40', '--batch', '128',\n",
            "                    '--out', str(WORK / 'resnet18_ssl.pt')],\n",
            "                   capture_output=True, text=True)\n",
            "print(r.stdout[-3000:]); print('STDERR:', r.stderr[-2000:])\n",
            "assert r.returncode == 0\n",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "%%time\n",
            "# verifier 学習: ssl init (評価は fold1 select → fold2 frozen)\n",
            "shutil.copytree(ver_dir, WORK / 'verifier_v1', dirs_exist_ok=True)\n",
            "r = subprocess.run([sys.executable, str(code_dir / 'train_verifier.py'),\n",
            "                    '--data', str(WORK / 'verifier_v1'),\n",
            "                    '--epochs', '8', '--init', f'ssl:{WORK}/resnet18_ssl.pt',\n",
            "                    '--out', str(WORK / 'verifier_ssl')],\n",
            "                   capture_output=True, text=True)\n",
            "print(r.stdout[-3000:]); print('STDERR:', r.stderr[-2000:])\n",
            "assert r.returncode == 0\n",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "%%time\n",
            "# 同環境の imagenet init 対照 (v70 値との一致も確認する)\n",
            "r = subprocess.run([sys.executable, str(code_dir / 'train_verifier.py'),\n",
            "                    '--data', str(WORK / 'verifier_v1'),\n",
            "                    '--epochs', '8', '--init', 'imagenet',\n",
            "                    '--out', str(WORK / 'verifier_imagenet')],\n",
            "                   capture_output=True, text=True)\n",
            "print(r.stdout[-3000:]); print('STDERR:', r.stderr[-2000:])\n",
            "assert r.returncode == 0\n",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "# 結果サマリ — 棄却ゲート判定\n",
            "ssl = json.load(open(WORK / 'verifier_ssl' / 'train_result.json'))\n",
            "img = json.load(open(WORK / 'verifier_imagenet' / 'train_result.json'))\n",
            "summary = {'ssl_fold2_auc': ssl.get('auc_fold2_verifier'),\n",
            "           'imagenet_fold2_auc': img.get('auc_fold2_verifier'),\n",
            "           'ssl_blend': ssl.get('auc_fold2_blend'),\n",
            "           'imagenet_blend': img.get('auc_fold2_blend'),\n",
            "           'conf_auc_fold2': ssl.get('auc_fold2_conf')}\n",
            "print(json.dumps(summary, indent=2))\n",
            "Path(WORK / 'v72_summary.json').write_text(json.dumps(\n",
            "    {'ssl': ssl, 'imagenet': img, 'summary': summary}, indent=2))\n",
        ],
    },
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                "language": "python"},
                 "language_info": {"name": "python"}},
    "cells": [{**c, "metadata": {},
               **({"outputs": [], "execution_count": None}
                  if c["cell_type"] == "code" else {})}
              for c in CELLS],
}

out = Path(__file__).resolve().parent.parent / "solar_filament_v72_gong_ssl.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out}")
