"""v75 — refiner flip-4 TTA + topology guard の OOF 評価ノートブック生成 (Colab)。

Fususu handoff (§2.29) の flip-4 TTA with topology guard を我々の reviewed
pipeline (prod refiner・fold2 OOF・frozen operating point) で検証する
推論のみの実験。学習なし — 同一 YOLO 提案・同一 refiner ckpt で
後処理 variant だけを変えた paired comparison:

| arm | prob map | 意味 |
|---|---|---|
| `baseline` | single-view (prod 経路) | sanity ≈ v74 0.444238 |
| `tta4` | flip4 平均 (guard なし) | TTA 効果そのもの |
| `tta4g` | flip4 平均 + topology guard | Fususu 型 guard の寄与 |

棄却/採用ゲート (事前固定, RESEARCH_LOG §2.30):
  - primary delta = tta4g − baseline @PQ0.45
    >= +0.005 → 昇格候補 (別 fold/LB で確認実験へ)
    [0, +0.005) → ノイズ棄却 / <0 → 棄却
  - secondary: tta4 − baseline で guard の load-bearing 度を分離
  - sanity: baseline PQ@0.45 が v74 実績 0.444238 ±0.001
  - report: flip4_guard の fallback 率 (Fususu は ~0.7%)

入力 (v74 と同一 reviewed セット、学習は一切行わない):
  - Drive solar_filament_v30/: train json, test_images, preproc, folds CSV
    (sha febb3f67...), baseline/YOLO ckpt
  - shingo257/solar-v73-code       : run_fold_oof_eval.py 等 (TTA variant は
    本 notebook が run_fold_oof_eval_tta.py を embedded 書き出し)
  - shingo257/solar-v73-assets     : v73_reviewed_provenance.json
"""

from __future__ import annotations

import json
from pathlib import Path

TTA_SCRIPT_PATH = Path(__file__).resolve().parent / "run_fold_oof_eval_tta.py"

MD_CELL = [
    "# v75 — refiner flip-4 TTA + topology guard 評価 (fold2 OOF, Colab)\n",
    "\n",
    "§2.29 で発見した Fususu handoff 由来の flip-4 TTA + topology guard を\n",
    "reviewed pipeline (prod refiner・fold2 OOF 235ファイル・frozen operating\n",
    "point conf0.26/minarea20/drop0.90) で検証する推論のみの実験。学習なし。\n",
    "\n",
    "| arm | prob map | mode |\n",
    "|---|---|---|\n",
    "| `baseline` | single-view (prod 経路) | none |\n",
    "| `tta4` | flip4 平均 | flip4 |\n",
    "| `tta4g` | flip4 平均 + topology guard (fallback→single) | flip4_guard |\n",
    "\n",
    "guard (crop 320²・0.5 二値): 面積比∈[0.85,1.15]・連結成分数差≤1・IoU≥0.85、\n",
    "違反時は single-view にフォールバック。\n",
    "\n",
    "**事前固定ゲート** (RESEARCH_LOG §2.30):\n",
    "- delta(tta4g − baseline)@PQ0.45 ≥ +0.005 → 昇格候補 (別fold/LB 確認へ)\n",
    "- [0,+0.005) → ノイズ棄却 / <0 → 棄却\n",
    "- sanity: baseline ≈ v74 実績 0.444238 ±0.001\n",
    "\n",
    "**Provenance pin:** `V73_PROVENANCE_SHA256` (v73/v74 と同一 manifest)。\n",
    "Drive `solar_filament_v35/kaggle.json` で kaggle 認証 (shingo257)。\n",
    "eval 終了時に Drive `solar_filament_v75/work/latest/` へ sha256 検証付きミラー。\n",
]

CELLS = [
    {"cell_type": "markdown", "source": MD_CELL},
    {
        "cell_type": "code",
        "source": [
            "import os, re, json, subprocess, sys, hashlib, shutil, time\n",
            "from pathlib import Path\n",
            "# Reviewed provenance pin is supplied externally — env var, Colab secret,\n",
            "# Kaggle secret — or declared interactively. Never derived from artifacts.\n",
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
            "# [0] Drive mount — primary data source (same pattern as all past colab runs)\n",
            "from google.colab import drive\n",
            "drive.mount('/content/drive')\n",
            "V30 = Path('/content/drive/MyDrive/solar_filament_v30')\n",
            "COMP = V30 / 'competition_data/MAGFiLO_1.0_Kaggle_2026'\n",
            "inputs = {\n",
            "    'train_json': COMP / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json',\n",
            "    'test_dir': COMP / 'test/test_images',\n",
            "    'preproc_dir': V30 / 'preprocessed_v30_q96',\n",
            "    'folds_csv': V30 / 'artifacts/grouped_folds.csv',\n",
            "    'baseline_ckpt': V30 / 'artifacts/mediseg_edgeattnet_refiner_fold2_resume.pt',\n",
            "    'yolo_ckpt': V30 / 'runs/yolo_fold_2/weights/best.pt',\n",
            "    'kaggle.json': Path('/content/drive/MyDrive/solar_filament_v35/kaggle.json'),\n",
            "}\n",
            "missing = {k: str(v) for k, v in inputs.items() if not v.exists()}\n",
            "if missing:\n",
            "    raise FileNotFoundError(f'Drive inputs missing: {missing}')\n",
            "train_json, test_dir, preproc_dir = (inputs['train_json'], inputs['test_dir'],\n",
            "                                     inputs['preproc_dir'])\n",
            "folds_csv, baseline_ckpt, yolo_ckpt = (inputs['folds_csv'], inputs['baseline_ckpt'],\n",
            "                                      inputs['yolo_ckpt'])\n",
            "if not any(test_dir.iterdir()):\n",
            "    raise RuntimeError('test_images dir is empty')\n",
            "EXPECTED_FOLDS_SHA = 'febb3f67237a058fa8a71f188fac262035048b55d860c0e4509514877f6af67e'\n",
            "actual = hashlib.sha256(folds_csv.read_bytes()).hexdigest()\n",
            "if actual != EXPECTED_FOLDS_SHA:\n",
            "    raise RuntimeError(f'fold CSV sha mismatch: {actual} (Drive artifacts/grouped_folds.csv)')\n",
            "print('folds_csv sha256 OK (true experimental split)')\n",
            "\n",
            "# [1] Kaggle auth + code/manifest download\n",
            "os.makedirs('/root/.kaggle', exist_ok=True)\n",
            "cred = inputs['kaggle.json'].read_bytes()\n",
            "print('auth: Drive kaggle.json')\n",
            "try:\n",
            "    _u = json.loads(cred)['username']\n",
            "except Exception:\n",
            "    _u = '?'\n",
            "print('kaggle user:', _u)\n",
            "Path('/root/.kaggle/kaggle.json').write_bytes(cred)\n",
            "os.chmod('/root/.kaggle/kaggle.json', 0o600)\n",
            "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U',\n",
            "                'kaggle', 'ultralytics', 'pycocotools'], check=True)\n",
            "\n",
            "WORK = Path('/content/work'); IN = WORK / 'input'; IN.mkdir(parents=True, exist_ok=True)\n",
            "def _dl(cmd):\n",
            "    r = subprocess.run(cmd, capture_output=True, text=True)\n",
            "    if r.returncode != 0:\n",
            "        raise RuntimeError('download failed rc=%s: %s\\n%s\\n%s' % (\n",
            "            r.returncode, cmd, r.stdout[-800:], r.stderr[-800:]))\n",
            "def _sha(p):\n",
            "    d = hashlib.sha256()\n",
            "    with open(p, 'rb') as h:\n",
            "        for b in iter(lambda: h.read(1 << 20), b''):\n",
            "            d.update(b)\n",
            "    return d.hexdigest()\n",
            "\n",
            "_dl(['kaggle', 'datasets', 'download', 'shingo257/solar-v73-code',\n",
            "     '-p', str(IN), '--unzip', '-q'])\n",
            "_dl(['kaggle', 'datasets', 'download', '-d', 'shingo257/solar-v73-assets',\n",
            "     '-f', 'v73_reviewed_provenance.json', '-p', str(IN)])\n",
            "\n",
            "code_dir = next(p.parent for p in IN.rglob('train_refiner_fold.py'))\n",
            "sys.path.insert(0, str(code_dir))\n",
            "manifest_path = next(IN.rglob('v73_reviewed_provenance.json'))\n",
            "from v73_split_provenance import (load_reviewed_manifest, load_fold_mapping,\n",
            "                                  validate_checkpoint_binding)\n",
            "manifest = load_reviewed_manifest(manifest_path, EXPECTED_PROVENANCE_SHA, 2)\n",
            "if manifest['folds_sha256'] != EXPECTED_FOLDS_SHA:\n",
            "    raise RuntimeError('reviewed split differs')\n",
            "coco = json.loads(train_json.read_text())\n",
            "load_fold_mapping(folds_csv, {i['file_name'] for i in coco['images']}, 2,\n",
            "                  manifest['folds_sha256'])\n",
            "validate_checkpoint_binding(yolo_ckpt, 'yolo', manifest)\n",
            "validate_checkpoint_binding(baseline_ckpt, 'baseline', manifest)\n",
            "\n",
            "# [2] TTA eval variant — embedded copy of repo scratch/run_fold_oof_eval_tta.py\n",
            "TTA_SCRIPT = ",
            "Path(code_dir / 'run_fold_oof_eval_tta.py').write_text(TTA_SCRIPT)\n",
            "import ast; ast.parse(TTA_SCRIPT)\n",
            "print('tta eval script written:', code_dir / 'run_fold_oof_eval_tta.py')\n",
            "for name, p in [('preproc', preproc_dir), ('folds', folds_csv),\n",
            "                ('baseline', baseline_ckpt), ('yolo', yolo_ckpt),\n",
            "                ('train_json', train_json), ('code', code_dir)]:\n",
            "    print(name, '->', p)\n",
            "import torch\n",
            "print('cuda:', torch.cuda.is_available(),\n",
            "      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')\n",
            "gpu_env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')\n",
            "\n",
            "MIRROR_DIR = Path('/content/drive/MyDrive/solar_filament_v75/work/latest')\n",
            "def mirror():\n",
            "    if not os.path.ismount('/content/drive'):\n",
            "        raise RuntimeError('Drive not mounted')\n",
            "    MIRROR_DIR.mkdir(parents=True, exist_ok=True)\n",
            "    digests = {}\n",
            "    for p in sorted(WORK.rglob('*')):\n",
            "        if p.is_file() and 'input' not in p.relative_to(WORK).parts:\n",
            "            dst = MIRROR_DIR / p.relative_to(WORK)\n",
            "            expected = _sha(p)\n",
            "            if dst.exists() and _sha(dst) == expected:\n",
            "                digests[str(p.relative_to(WORK))] = expected; continue\n",
            "            dst.parent.mkdir(parents=True, exist_ok=True)\n",
            "            shutil.copy2(p, dst)\n",
            "            if _sha(dst) != expected:\n",
            "                raise RuntimeError(f'Drive mirror sha mismatch: {p.name}')\n",
            "            digests[str(p.relative_to(WORK))] = expected\n",
            "            print('mirrored', p.name, p.stat().st_size)\n",
            "    (MIRROR_DIR / 'sha256_manifest.json').write_text(json.dumps(digests, indent=1))\n",
            "    print('mirror DONE ->', MIRROR_DIR)\n",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "%%time\n",
            "# [eval] 3 arm — 同一 YOLO 提案・同一 refiner で TTA variant だけを変える\n",
            "r = subprocess.run([sys.executable, str(code_dir / 'run_fold_oof_eval_tta.py'),\n",
            "                    '--train-json', str(train_json),\n",
            "                    '--test-dir', str(test_dir),\n",
            "                    '--preproc-dir', str(preproc_dir),\n",
            "                    '--folds-csv', str(folds_csv), '--fold', '2',\n",
            "                    '--provenance-manifest', str(manifest_path),\n",
            "                    '--provenance-sha256', EXPECTED_PROVENANCE_SHA,\n",
            "                    '--yolo', str(yolo_ckpt), '--model', 'edgeatt',\n",
            "                    '--refiner', f'baseline={baseline_ckpt}:none',\n",
            "                    '--refiner', f'tta4={baseline_ckpt}:flip4',\n",
            "                    '--refiner', f'tta4g={baseline_ckpt}:flip4_guard',\n",
            "                    '--out', str(WORK / 'v75_eval.json')],\n",
            "                   capture_output=True, text=True, env=gpu_env)\n",
            "print(r.stdout[-4000:]); print('STDERR:', r.stderr[-2000:])\n",
            "if r.returncode != 0 or not (WORK / 'v75_eval.json').exists():\n",
            "    raise RuntimeError(f'eval failed rc={r.returncode}')\n",
            "mirror()\n",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "# 結果サマリ — 事前固定ゲート判定 (§2.30)\n",
            "rep = json.load(open(WORK / 'v75_eval.json'))\n",
            "arms = rep['arms']\n",
            "pq = lambda a: arms[a]['pq_by_threshold']['0.45']\n",
            "print(json.dumps({a: r['pq_by_threshold'] for a, r in arms.items()}, indent=2))\n",
            "d_g = round(pq('tta4g') - pq('baseline'), 6)\n",
            "d_u = round(pq('tta4') - pq('baseline'), 6)\n",
            "fb = arms['tta4g'].get('tta_fallback') or {}\n",
            "gate = {\n",
            "  'baseline_pq_0.45': pq('baseline'),\n",
            "  'sanity_baseline_near_v74': abs(pq('baseline') - 0.444238) < 0.001,\n",
            "  'baseline_diff_vs_v74': round(pq('baseline') - 0.444238, 6),\n",
            "  'tta4_pq': pq('tta4'), 'tta4g_pq': pq('tta4g'),\n",
            "  'delta_tta4g_minus_baseline': d_g,\n",
            "  'delta_tta4_minus_baseline': d_u,\n",
            "  'guard_stats': fb,\n",
            "  'decision': ('promote_candidate' if d_g >= 0.005\n",
            "               else 'reject' if d_g < 0 else 'reject_noise'),\n",
            "}\n",
            "print(json.dumps(gate, indent=2))\n",
            "Path(WORK / 'v75_summary.json').write_text(json.dumps(\n",
            "    {'eval': rep, 'gate': gate}, indent=2))\n",
            "mirror()\n",
        ],
    },
]


def build_notebook(script_text: str) -> dict:
    cells = json.loads(json.dumps(CELLS))
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        for i, line in enumerate(cell["source"]):
            if line.strip() == "TTA_SCRIPT =":
                cell["source"][i] = "TTA_SCRIPT = " + repr(script_text) + "\n"
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                    "language": "python"},
                     "language_info": {"name": "python"}},
        "cells": [{**c, "metadata": {},
                   **({"outputs": [], "execution_count": None}
                      if c["cell_type"] == "code" else {})}
                  for c in cells],
    }
    return nb


def main() -> Path:
    nb = build_notebook(TTA_SCRIPT_PATH.read_text(encoding="utf-8"))
    out = (Path(__file__).resolve().parent.parent
           / "solar_filament_v75_tta_guard_colab.ipynb")
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()
