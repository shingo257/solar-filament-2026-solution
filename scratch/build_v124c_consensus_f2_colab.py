"""v124c — cons2of3_c26 の fold-2 honest 複製 eval (Colab、§2.99 補完)。

v120 の fold-2 +0.01155 は v116 crop det が fold-2 eval 画像を学習していた
ため撤回済み (§2.98/§2.99 開示)。本ノートは **v121c (fold-0) ゲート通過
時のみ**実行する第二 fold 確認: crop det を v124a honest 版
(folds(0,1) のみ学習 = `runs/v124_croptrain_f2_honest`) に差し替え、
v120 と全く同じ 6 arm・評価経路・採点で fold-2 OOF を再測定する。

detector 3 本 (いずれも folds(0,1) 学習 → fold-2 OOF honest):
- `yolo`: prod yolo11m fold_2 (manifest role `yolo` binding、fold=2)
- `crop`: v124a croptrain fold2 honest 版 (fit_files のみ、trusted-pin
  = v124a manifest 記録 `datasets.croptrain.weights_sha256`)
- `yolo26`: v117 yolo26m fold2 (trusted-pin = v117 manifest 記録
  `results.yolo26m.weights_sha256` — v120 と同一 byte)

判定 (§2.99 補完、事前固定):
  - cons2of3_c26 Δ@0.45 (vs fold-2 prod arm) >= +0.005
    → `second_fold_positive` (fold-0 と合わせ 2 fold で効果確認)
    未満 → `second_fold_negative` (fold-0 陽性が再現しなかった)
  - sanity: prod arm PQ@0.45 が fold-2 prod 既知値 0.444238 ±0.001
  - 片 fold 陽性だけでは提出差替えの根拠にしない (§2.99 規約)

入力 (reviewed セット + v124a/v117 detector ckpt、学習なし):
  - Drive solar_filament_v30/: train json, test_images, preproc, folds CSV
    (sha febb3f67...), baseline refiner ckpt, yolo/crop/yolo26 best.pt,
    artifacts/v124a_*・v117_*_colab_manifest.json
  - shingo257/solar-v73-code       : shared helpers (eval script は embedded)
  - shingo257/solar-v73-assets     : v73_reviewed_provenance.json (fold-2)
"""

from __future__ import annotations

import json
from pathlib import Path

CONSENSUS_SCRIPT_PATH = (Path(__file__).resolve().parent
                         / "run_fold_oof_eval_consensus.py")

MD_CELL = [
    "# v124c — cons2of3_c26 fold-2 honest 複製 eval (Colab、条件付き)\n",
    "\n",
    "v120 の fold-2 +0.01155 は crop det の漏洩で撤回済み (§2.98 開示)。\n",
    "**v121c (fold-0) ゲート通過時のみ実行**する第二 fold 確認 — crop det を\n",
    "v124a honest 版 (folds(0,1) のみ学習) に差し替え、v120 と同一の 6 arm・\n",
    "評価経路で fold-2 OOF を再測定:\n",
    "\n",
    "| arm | 放出ルール |\n",
    "|---|---|\n",
    "| `prod` | yolo 提案のみ conf>=0.26 (sanity ≈0.444238) |\n",
    "| `union3` | 全クラスタ (union 参照) |\n",
    "| `union3_c26` | max conf>=0.26 のクラスタ |\n",
    "| `cons2of3` | >=2 detector 支持 (低conf合意救済) |\n",
    "| `cons2of3_c26` | conf>=0.26 の支持 >=2 (複製対象 arm) |\n",
    "| `cons3of3` | 全員支持 |\n",
    "\n",
    "**判定** (§2.99 補完):\n",
    "- cons2of3_c26 Δ@0.45 ≥ +0.005 → `second_fold_positive`\n",
    "- 未満 → `second_fold_negative` (fold-0 陽性は未再現)\n",
    "- sanity: prod arm ≈ v76 実績 0.444238 ±0.001\n",
    "- 片 fold 陽性だけでは提出差替えの根拠にしない\n",
    "\n",
    "**Provenance pin:** `V73_PROVENANCE_SHA256` (v73-v120 と同一 fold-2\n",
    "manifest)。yolo det は manifest binding、crop/yolo26 det は各学習\n",
    "manifest 記録の weights_sha256 を trusted-pin として受理。\n",
    "refiner は prod baseline のみ (manifest role=baseline)。\n",
    "Drive `solar_filament_v35/kaggle.json` で kaggle 認証 (shingo257)。\n",
    "eval 終了時に Drive `solar_filament_v124/work/latest/` へ sha256 検証付きミラー。\n",
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
            "    'crop_ckpt': V30 / 'runs/v124_croptrain_f2_honest/yolo_croptrain_fold2/weights/best.pt',\n",
            "    'yolo26_ckpt': V30 / 'runs/v117_yolo26/yolo_yolo26m_fold2/weights/best.pt',\n",
            "    'v124a_manifest': V30 / 'artifacts/v124a_croptrain_f2_colab_manifest.json',\n",
            "    'v117_manifest': V30 / 'artifacts/v117_yolo26_colab_manifest.json',\n",
            "    'kaggle.json': Path('/content/drive/MyDrive/solar_filament_v35/kaggle.json'),\n",
            "}\n",
            "missing = {k: str(v) for k, v in inputs.items() if not v.exists()}\n",
            "if missing:\n",
            "    raise FileNotFoundError(f'Drive inputs missing: {missing}')\n",
            "train_json, test_dir, preproc_dir = (inputs['train_json'], inputs['test_dir'],\n",
            "                                     inputs['preproc_dir'])\n",
            "folds_csv, baseline_ckpt, yolo_ckpt = (inputs['folds_csv'], inputs['baseline_ckpt'],\n",
            "                                      inputs['yolo_ckpt'])\n",
            "crop_ckpt, yolo26_ckpt = inputs['crop_ckpt'], inputs['yolo26_ckpt']\n",
            "# §2.99 — crop/yolo26 fold-2 重みの trusted-pin は各学習 manifest が正本\n",
            "# (学習時に記録された weights_sha256 を読み取り、eval cell で byte 照合)\n",
            "_v124a_man = json.loads(inputs['v124a_manifest'].read_text(encoding='utf-8'))\n",
            "_v117_man = json.loads(inputs['v117_manifest'].read_text(encoding='utf-8'))\n",
            "TRUSTED_SHA256 = {\n",
            "    str(crop_ckpt): _v124a_man['datasets']['croptrain']['weights_sha256'],\n",
            "    str(yolo26_ckpt): _v117_man['results']['yolo26m']['weights_sha256'],\n",
            "}\n",
            "print('trusted pins from training manifests:',\n",
            "      {Path(k).parent.parent.parent.name: v[:12] for k, v in TRUSTED_SHA256.items()})\n",
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
            "# [2] consensus eval variant — embedded copy of repo scratch/run_fold_oof_eval_consensus.py (§2.98 派生・fold-2 honest 再実行)\n",
            "CONSENSUS_SCRIPT = ",
            "Path(code_dir / 'run_fold_oof_eval_consensus.py').write_text(CONSENSUS_SCRIPT)\n",
            "import ast; ast.parse(CONSENSUS_SCRIPT)\n",
            "print('consensus eval script written:', code_dir / 'run_fold_oof_eval_consensus.py')\n",
            "for name, p in [('preproc', preproc_dir), ('folds', folds_csv),\n",
            "                ('baseline', baseline_ckpt), ('yolo', yolo_ckpt),\n",
            "                ('crop', crop_ckpt), ('yolo26', yolo26_ckpt),\n",
            "                ('train_json', train_json), ('code', code_dir)]:\n",
            "    print(name, '->', p)\n",
            "import torch\n",
            "print('cuda:', torch.cuda.is_available(),\n",
            "      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')\n",
            "gpu_env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')\n",
            "\n",
            "MIRROR_DIR = Path('/content/drive/MyDrive/solar_filament_v124/work/latest')\n",
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
            "# [eval] 6 arm — fold-2 OOF で 3 detector の提案クラスタから合意数で放出 (§2.99 補完)\n",
            "# crop/yolo26 det の byte-pin を実行時に照合 (trusted-sha256 受理の前提)\n",
            "for path_str, expected in TRUSTED_SHA256.items():\n",
            "    p = Path(path_str)\n",
            "    actual = _sha(p)\n",
            "    if actual != expected:\n",
            "        raise RuntimeError(f'trusted sha256 mismatch: {p} {actual[:16]} != {expected[:16]}')\n",
            "    print('trusted sha256 OK:', p, actual[:16])\n",
            "trusted_args = [x for sha in TRUSTED_SHA256.values() for x in ('--trusted-sha256', sha)]\n",
            "r = subprocess.run([sys.executable, str(code_dir / 'run_fold_oof_eval_consensus.py'),\n",
            "                    '--train-json', str(train_json),\n",
            "                    '--test-dir', str(test_dir),\n",
            "                    '--preproc-dir', str(preproc_dir),\n",
            "                    '--folds-csv', str(folds_csv), '--fold', '2',\n",
            "                    '--provenance-manifest', str(manifest_path),\n",
            "                    '--provenance-sha256', EXPECTED_PROVENANCE_SHA,\n",
            "                    '--detector', f'yolo={yolo_ckpt}',\n",
            "                    '--detector', f'crop={crop_ckpt}',\n",
            "                    '--detector', f'yolo26={yolo26_ckpt}',\n",
            "                    '--model', 'edgeatt',\n",
            "                    *trusted_args,\n",
            "                    '--refiner', f'baseline={baseline_ckpt}',\n",
            "                    '--out', str(WORK / 'v124_eval.json')],\n",
            "                   capture_output=True, text=True, env=gpu_env)\n",
            "print(r.stdout[-4000:]); print('STDERR:', r.stderr[-2000:])\n",
            "if r.returncode != 0 or not (WORK / 'v124_eval.json').exists():\n",
            "    raise RuntimeError(f'eval failed rc={r.returncode}')\n",
            "mirror()\n",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "# 結果サマリ — 第二 fold 判定 (§2.99 補完: cons2of3_c26 の fold-2 clean 複製)\n",
            "rep = json.load(open(WORK / 'v124_eval.json'))\n",
            "arms = rep['arms']\n",
            "pq = lambda a: arms[a]['pq_by_threshold']['0.45']\n",
            "print(json.dumps({a: r['pq_by_threshold'] for a, r in arms.items()}, indent=2))\n",
            "print('emitted/image:', rep.get('emitted_per_arm'))\n",
            "base = pq('baseline::prod')\n",
            "deltas = {a: round(pq(a) - base, 6) for a in arms if a != 'baseline::prod'}\n",
            "rep_delta = deltas['baseline::cons2of3_c26']  # §2.99 の複製対象 arm\n",
            "gate = {\n",
            "  'prod_pq_0.45': base,\n",
            "  'sanity_prod_near_v76': abs(base - 0.444238) < 0.001,\n",
            "  'prod_diff_vs_v76': round(base - 0.444238, 6),\n",
            "  'delta_by_arm': deltas,\n",
            "  'cons2of3_c26_delta_fold2': rep_delta,\n",
            "  'fold0_reference': 'v121_summary.json (fold-0 結果は別途参照)',\n",
            "  'v120_retracted_delta': 0.01155,  # 漏洩のため撤回済み — 参照のみ\n",
            "  'decision': ('second_fold_positive' if rep_delta >= 0.005\n",
            "               else 'second_fold_negative'),\n",
            "}\n",
            "print(json.dumps(gate, indent=2))\n",
            "Path(WORK / 'v124_summary.json').write_text(json.dumps(\n",
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
            if line.strip() == "CONSENSUS_SCRIPT =":
                cell["source"][i] = "CONSENSUS_SCRIPT = " + repr(script_text) + "\n"
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
    nb = build_notebook(CONSENSUS_SCRIPT_PATH.read_text(encoding="utf-8"))
    out = (Path(__file__).resolve().parent.parent
           / "solar_filament_v124c_consensus_f2_colab.ipynb")
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()
