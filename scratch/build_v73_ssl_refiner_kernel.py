"""v73 — GONG SSL → refiner encoder init 実験の Kaggle ノートブックを生成する。

v70/v72 の残存仮説「小データで画像特徴が学べない」に対し、無ラベル GONG
全日面コーパス (1,889 フレーム、test ±6h 除外済み、sha `corpus manifest`) で
MediSeg U-Net エンコーダ (inc/down1-4) を SimCLR 事前学習し、prod refiner
(`MediSegEdgeAttUNet`) の fold2 再学習初期値にする。RQ 側 (verifier) ではなく
SQ 側 (マスク品質) を狙う唯一残ったレーン。

棄却/採用ゲート (事前固定, RESEARCH_LOG §2.26):
  - fold2 OOF PQ (prod operating point: conf 0.26 / thr 0.45 / min_area 20 /
    drop_contained 0.90) で ssl arm − scratch arm が >= +0.010 → 別 fold・別
    seed で確認実験へ進む
  - 0 <= delta < +0.010 → ノイズ帯。追加確認なしでは棄却
  - delta < 0 → 棄却
  同時に baseline ckpt (mediseg_edgeattnet_refiner_fold2_resume.pt) を同一
  提案・同一後処理で採点し ~0.448 の再現を sanity ゲートとする。

データセット (private):
  - shingo257/solar-gong-ssl              : corpus batch_*.npz (1,889 frames)
  - shingo257/solar-filament-v35-artifacts: preprocessed_v30_q96/*.jpeg
  - shingo257/solar-filament-2026-weights : yolo_fold2_best.pt
  - shingo257/solar-v73-assets            : grouped_folds_train_split.csv
      (SHA256 febb3f67... = OOF cache と一致する実験分割) +
      mediseg_edgeattnet_refiner_fold2_resume.pt (baseline 参照)
  - shingo257/solar-v73-code              : mediseg_models.py, ownership.py,
      pretrain_unet_encoder_ssl.py, train_refiner_fold.py,
      run_fold_oof_eval.py, v73_split_provenance.py, build_gong_ssl_corpus.py
  - competition_sources: filament-segmentation-2026 (train annotations)
"""

from __future__ import annotations

import json
from pathlib import Path

CELLS = [
    {
        "cell_type": "markdown",
        "source": [
            "# v73 — GONG SSL → refiner encoder init (fold2 screen)\n",
            "\n",
            "**Provenance gate:** corpus rebuilt under `train-test-temporal-v2`\n",
            "with `corpus_config.json`; split CSV + YOLO/baseline checkpoint bytes\n",
            "must be bound by an independently reviewed manifest (`v73_reviewed_provenance.json`,\n",
            "set the externally reviewed SHA256 as `V73_PROVENANCE_SHA256`).\n",
            "残る未試行レーン: 無ラベル GONG 全日面コーパスで **セグメンテーション側**\n",
            "エンコーダを SimCLR 事前学習し、prod EdgeAtt refiner の fold2 OOF PQ が\n",
            "scratch init を上回るかを検証する。\n",
            "\n",
            "**事前固定ゲート** (RESEARCH_LOG §2.26):\n",
            "- delta(PQ) = ssl − scratch >= +0.010 → 確認実験 (fold0/1 + seed) へ\n",
            "- 0 <= delta < +0.010 → 棄却 (ノイズ帯)\n",
            "- delta < 0 → 棄却\n",
            "- sanity: baseline ckpt の fold2 PQ が ~0.448 から外れすぎないこと\n",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "import os, shutil, subprocess, sys, json\n",
            "from pathlib import Path\n",
            "import re\n",
            "EXPECTED_PROVENANCE_SHA = os.environ.get('V73_PROVENANCE_SHA256', '')\n",
            "if not EXPECTED_PROVENANCE_SHA:\n",
            "    try:\n",
            "        from google.colab import userdata\n",
            "        EXPECTED_PROVENANCE_SHA = userdata.get('V73_PROVENANCE_SHA256') or ''\n",
            "    except Exception:\n",
            "        pass\n",
            "if not EXPECTED_PROVENANCE_SHA:\n",
            "    try:\n",
            "        from kaggle_secrets import UserSecretsClient\n",
            "        EXPECTED_PROVENANCE_SHA = UserSecretsClient().get_secret('V73_PROVENANCE_SHA256')\n",
            "    except Exception:\n",
            "        pass\n",
            "if not re.fullmatch(r'[0-9a-f]{64}', EXPECTED_PROVENANCE_SHA):\n",
            "    try:\n",
            "        EXPECTED_PROVENANCE_SHA = input('V73_PROVENANCE_SHA256 (reviewed manifest hash): ').strip()\n",
            "    except (EOFError, OSError):\n",
            "        pass\n",
            "if not re.fullmatch(r'[0-9a-f]{64}', EXPECTED_PROVENANCE_SHA):\n",
            "    raise RuntimeError('Set V73_PROVENANCE_SHA256 after independent provenance review')\n",
            "\n",
            "IN = Path('/kaggle/input')\n",
            "print(sorted(str(p.relative_to(IN)) for p in IN.rglob('*') if p.is_dir())[:40])\n",
            "corpus_dir = next(p.parent for p in IN.rglob('batch_*.npz'))\n",
            "preproc_dir = next(p.parent for p in IN.rglob('preprocessed_v30_q96/*.jpeg'))\n",
            "folds_csv = next(IN.rglob('grouped_folds_train_split.csv'))\n",
            "import hashlib\n",
            "EXPECTED_FOLDS_SHA = 'febb3f67237a058fa8a71f188fac262035048b55d860c0e4509514877f6af67e'\n",
            "actual = hashlib.sha256(folds_csv.read_bytes()).hexdigest()\n",
            "if actual != EXPECTED_FOLDS_SHA:\n",
            "    raise RuntimeError(f'fold CSV sha mismatch: {actual}')\n",
            "print('folds_csv sha256 OK (true experimental split)')\n",
            "baseline_ckpt = next(IN.rglob('mediseg_edgeattnet_refiner_fold2_resume.pt'))\n",
            "yolo_ckpt = next(IN.rglob('yolo_fold2_best.pt'))\n",
            "train_json = next(IN.rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json'))\n",
            "code_dir = next(p.parent for p in IN.rglob('train_refiner_fold.py'))\n",
            "sys.path.insert(0, str(code_dir))\n",
            "# Provenance is a reviewed input, never auto-generated from ambiguous legacy weights.\n",
            "manifest_path = next(IN.rglob('v73_reviewed_provenance.json'))\n",
            "from v73_split_provenance import load_reviewed_manifest, load_fold_mapping, validate_checkpoint_binding\n",
            "manifest = load_reviewed_manifest(manifest_path, EXPECTED_PROVENANCE_SHA, 2)\n",
            "if manifest['folds_sha256'] != EXPECTED_FOLDS_SHA:\n",
            "    raise RuntimeError('reviewed split differs')\n",
            "test_dirs = [p for p in IN.rglob('test_images') if p.is_dir()]\n",
            "if len(test_dirs) != 1:\n",
            "    raise RuntimeError('exactly one actual competition test_images directory required')\n",
            "test_dir = test_dirs[0]\n",
            "coco = json.loads(train_json.read_text())\n",
            "load_fold_mapping(folds_csv, {i['file_name'] for i in coco['images']}, 2, manifest['folds_sha256'])\n",
            "validate_checkpoint_binding(yolo_ckpt, 'yolo', manifest)\n",
            "validate_checkpoint_binding(baseline_ckpt, 'baseline', manifest)\n",
            "# No package installs, CUDA inspection or training until all artifact preflights pass.\n",
            "subprocess.run([sys.executable, str(code_dir / 'pretrain_unet_encoder_ssl.py'),\n",
            "                '--corpus', str(corpus_dir), '--train-json', str(train_json),\n",
            "                '--test-dir', str(test_dir), '--validate-only',\n",
            "                '--out', '/kaggle/working/unet_encoder_ssl.pt'], check=True)\n",
            "WORK = Path('/kaggle/working')\n",
            "for name, p in [('corpus', corpus_dir), ('preproc', preproc_dir),\n",
            "                ('folds', folds_csv), ('baseline', baseline_ckpt),\n",
            "                ('yolo', yolo_ckpt), ('train_json', train_json),\n",
            "                ('code', code_dir)]:\n",
            "    print(name, '->', p)\n",
            "import torch\n",
            "print('cuda:', torch.cuda.is_available(),\n",
            "      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')\n",
            "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',\n",
            "                'ultralytics', 'pycocotools'], check=True)\n",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "%%time\n",
            "# [1/4] GONG corpus で U-Net encoder を SimCLR 事前学習 (solar3 変換込み)\n",
            "gpu_env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')\n",
            "r = subprocess.run([sys.executable,\n",
            "                    str(code_dir / 'pretrain_unet_encoder_ssl.py'),\n",
            "                    '--corpus', str(corpus_dir),\n",
            "                    '--train-json', str(train_json),\n",
            "                    '--test-dir', str(test_dir),\n",
            "                    '--epochs', '60', '--batch', '32', '--crop', '320',\n",
            "                    '--out', str(WORK / 'unet_encoder_ssl.pt')],\n",
            "                   capture_output=True, text=True, env=gpu_env)\n",
            "print(r.stdout[-3000:]); print('STDERR:', r.stderr[-2000:])\n",
            "if r.returncode != 0 or not (WORK / 'unet_encoder_ssl.pt').exists():\n",
            "    raise RuntimeError(f'SSL pretrain failed rc={r.returncode}')\n",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "%%time\n",
            "# [2/4] fold2 refiner 学習 (folds 0+1 で学習、fold2 valid) — scratch arm\n",
            "r = subprocess.run([sys.executable, str(code_dir / 'train_refiner_fold.py'),\n",
            "                    '--train-json', str(train_json),\n",
            "                    '--test-dir', str(test_dir),\n",
            "                    '--preproc-dir', str(preproc_dir),\n",
            "                    '--folds-csv', str(folds_csv), '--fold', '2',\n",
            "                    '--expected-folds-sha256', manifest['folds_sha256'],\n",
            "                    '--model', 'edgeatt', '--init', 'scratch',\n",
            "                    '--epochs', '14', '--seed', '3407',\n",
            "                    '--out', str(WORK / 'edgeatt_scratch_fold2.pt')],\n",
            "                   capture_output=True, text=True, env=gpu_env)\n",
            "print(r.stdout[-3000:]); print('STDERR:', r.stderr[-2000:])\n",
            "if r.returncode != 0 or not (WORK / 'edgeatt_scratch_fold2.pt').exists():\n",
            "    raise RuntimeError(f'scratch arm failed rc={r.returncode}')\n",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "%%time\n",
            "# [3/4] 同一レシピ・同一 seed — ssl encoder init arm\n",
            "if not (WORK / 'unet_encoder_ssl.pt').exists():\n",
            "    raise RuntimeError('SSL encoder checkpoint missing — pretrain cell failed')\n",
            "r = subprocess.run([sys.executable, str(code_dir / 'train_refiner_fold.py'),\n",
            "                    '--train-json', str(train_json),\n",
            "                    '--test-dir', str(test_dir),\n",
            "                    '--preproc-dir', str(preproc_dir),\n",
            "                    '--folds-csv', str(folds_csv), '--fold', '2',\n",
            "                    '--expected-folds-sha256', manifest['folds_sha256'],\n",
            "                    '--model', 'edgeatt',\n",
            "                    '--init', f'ssl:{WORK}/unet_encoder_ssl.pt',\n",
            "                    '--epochs', '14', '--seed', '3407',\n",
            "                    '--out', str(WORK / 'edgeatt_ssl_fold2.pt')],\n",
            "                   capture_output=True, text=True, env=gpu_env)\n",
            "print(r.stdout[-3000:]); print('STDERR:', r.stderr[-2000:])\n",
            "if r.returncode != 0 or not (WORK / 'edgeatt_ssl_fold2.pt').exists():\n",
            "    raise RuntimeError(f'ssl arm failed rc={r.returncode}')\n",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "%%time\n",
            "# [4/4] fold2 OOF eval — 3 arm を同一 YOLO 提案・同一後処理で採点\n",
            "for _p in ('edgeatt_scratch_fold2.pt', 'edgeatt_ssl_fold2.pt'):\n",
            "    if not (WORK / _p).exists():\n",
            "        raise RuntimeError(f'refiner checkpoint missing: {_p}')\n",
            "r = subprocess.run([sys.executable, str(code_dir / 'run_fold_oof_eval.py'),\n",
            "                    '--train-json', str(train_json),\n",
            "                    '--test-dir', str(test_dir),\n",
            "                    '--preproc-dir', str(preproc_dir),\n",
            "                    '--folds-csv', str(folds_csv), '--fold', '2',\n",
            "                    '--provenance-manifest', str(manifest_path),\n",
            "                    '--provenance-sha256', EXPECTED_PROVENANCE_SHA,\n",
            "                    '--yolo', str(yolo_ckpt), '--model', 'edgeatt',\n",
            "                    '--refiner', f'baseline={baseline_ckpt}',\n",
            "                    '--refiner', f'scratch={WORK}/edgeatt_scratch_fold2.pt',\n",
            "                    '--refiner', f'ssl={WORK}/edgeatt_ssl_fold2.pt',\n",
            "                    '--out', str(WORK / 'v73_eval.json')],\n",
            "                   capture_output=True, text=True, env=gpu_env)\n",
            "print(r.stdout[-4000:]); print('STDERR:', r.stderr[-2000:])\n",
            "if r.returncode != 0 or not (WORK / 'v73_eval.json').exists():\n",
            "    raise RuntimeError(f'eval failed rc={r.returncode}')\n",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "# 結果サマリ — 事前固定ゲート判定\n",
            "rep = json.load(open(WORK / 'v73_eval.json'))\n",
            "arms = rep['arms']\n",
            "table = {a: r['pq_by_threshold'] for a, r in arms.items()}\n",
            "print(json.dumps(table, indent=2))\n",
            "gate = {\n",
            "  'baseline_pq_0.45': arms['baseline']['pq_by_threshold']['0.45'],\n",
            "  'scratch_pq_0.45': arms['scratch']['pq_by_threshold']['0.45'],\n",
            "  'ssl_pq_0.45': arms['ssl']['pq_by_threshold']['0.45'],\n",
            "  'delta_ssl_minus_scratch': round(\n",
            "      arms['ssl']['pq_by_threshold']['0.45']\n",
            "      - arms['scratch']['pq_by_threshold']['0.45'], 6),\n",
            "  'sanity_baseline_near_0.448':\n",
            "      abs(arms['baseline']['pq_by_threshold']['0.45'] - 0.4483) < 0.01,\n",
            "}\n",
            "print(json.dumps(gate, indent=2))\n",
            "Path(WORK / 'v73_summary.json').write_text(json.dumps(\n",
            "    {'eval': rep, 'gate': gate}, indent=2))\n",
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

out = Path(__file__).resolve().parent.parent / "solar_filament_v73_ssl_refiner.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out}")
