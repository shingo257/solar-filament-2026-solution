"""v120 — 検出器間コンセンサス・ゲートの OOF 評価ノートブック生成 (Colab)。

v115 (§2.91) で union_yolo_mrcnn は FN −230 を救ったが FP が 750→2745 に
氾濫して −0.097。逆方向 — 複数検出器の**合意でのみ提案を放出** — は
一度も試されていない (§2.98)。学習なし・推論のみ:

| arm | 放出ルール |
|---|---|
| `prod` | yolo 提案のみ conf>=0.26 (baseline sanity) |
| `union3` | 全クラスタ (union 参照・既知の壊滅確認) |
| `union3_c26` | max member conf>=0.26 のクラスタ (v115 型 union) |
| `cons2of3` | >=2 detector が支持 (floor 合意による低conf救済) |
| `cons2of3_c26` | conf>=0.26 の支持が >=2 detector (純粋合意) |
| `cons3of3` | 3 detector 全員が支持 |

detector 3 本 (いずれも folds(0,1) 学習 → fold-2 OOF honest):
- `yolo`: prod yolo11m (manifest role `yolo` binding)
- `crop`: v116 croptrain (trusted byte-pin、§2.98 開示)
- `yolo26`: v117 yolo26m (trusted byte-pin、§2.98 開示)

提案を mask-IoU>=0.5 の推移クラスタ (union-find、cross-detector edge
のみ) にまとめ arm ごとに放出 → 同一 edgeatt refiner (baseline) →
同一後処理・同一 PQ 採点。

棄却/採用ゲート (事前固定, RESEARCH_LOG §2.98):
  - いずれか arm で PQ(arm,0.45) − PQ(prod,0.45) >= +0.010
    → 昇格候補 (fold0 honest 複製へ、crop/yolo26 再学習 ~8h)
    未満 → 棄却。fold2 単独では提出不変。
  - sanity: prod arm PQ@0.45 が v76 実績 0.444238 ±0.001

入力 (reviewed セット + v116/v117 detector ckpt、学習なし):
  - Drive solar_filament_v30/: train json, test_images, preproc, folds CSV
    (sha febb3f67...), baseline refiner ckpt, yolo/crop/yolo26 best.pt
  - shingo257/solar-v73-code       : shared helpers (eval script は embedded)
  - shingo257/solar-v73-assets     : v73_reviewed_provenance.json
"""

from __future__ import annotations

import json
from pathlib import Path

CONSENSUS_SCRIPT_PATH = (Path(__file__).resolve().parent
                         / "run_fold_oof_eval_consensus.py")

MD_CELL = [
    "# v120 — 検出器間コンセンサス・ゲート評価 (fold2 OOF, Colab)\n",
    "\n",
    "v115 で union は FN −230 を救ったが FP 3.7 倍で壊滅。逆方向 —\n",
    "複数検出器の**合意でのみ提案を放出** — を試す推論のみの実験 (§2.98)。\n",
    "detector 3 本 (prod yolo11m・v116 croptrain・v117 yolo26m、いずれも\n",
    "folds(0,1) 学習 → fold-2 honest) の提案を mask-IoU>=0.5 でクラスタ化し、\n",
    "合意数で放出判定:\n",
    "\n",
    "| arm | 放出ルール |\n",
    "|---|---|\n",
    "| `prod` | yolo 提案のみ conf>=0.26 (sanity ≈0.444238) |\n",
    "| `union3` | 全クラスタ (union 参照) |\n",
    "| `union3_c26` | max conf>=0.26 のクラスタ (v115 型) |\n",
    "| `cons2of3` | >=2 detector 支持 (低conf合意救済) |\n",
    "| `cons2of3_c26` | conf>=0.26 の支持 >=2 (純粋合意) |\n",
    "| `cons3of3` | 全員支持 |\n",
    "\n",
    "**事前固定ゲート** (RESEARCH_LOG §2.98):\n",
    "- いずれか arm で Δ@0.45 ≥ +0.010 → 昇格候補 (fold0 複製へ)\n",
    "- 未満 → 棄却。fold2 単独では提出不変\n",
    "- sanity: prod arm ≈ v76 実績 0.444238 ±0.001\n",
    "\n",
    "**Provenance pin:** `V73_PROVENANCE_SHA256` (v73-v119 と同一 manifest)。\n",
    "yolo det は manifest binding、crop/yolo26 det は --trusted-sha256\n",
    "byte-pin (pin 値は §2.98 開示) で受理。refiner は prod baseline のみ\n",
    "(manifest role=baseline)。\n",
    "Drive `solar_filament_v35/kaggle.json` で kaggle 認証 (shingo257)。\n",
    "eval 終了時に Drive `solar_filament_v120/work/latest/` へ sha256 検証付きミラー。\n",
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
            "    'crop_ckpt': V30 / 'runs/v116_croptrain/yolo_croptrain_fold2/weights/best.pt',\n",
            "    'yolo26_ckpt': V30 / 'runs/v117_yolo26/yolo_yolo26m_fold2/weights/best.pt',\n",
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
            "# §2.98 事前固定 pin — v116/v117 detector は manifest 非宣言のため byte-pin で受理\n",
            "# (両方 best.pt 同名なので name ではなく full path で対応付ける)\n",
            "TRUSTED_SHA256 = {\n",
            "    str(inputs['crop_ckpt']): 'c23b1bb79620c62fd85984d032310bae26c240fd1be0aaf7df364fbc560fbcda',\n",
            "    str(inputs['yolo26_ckpt']): '0c8bb71f6ab4398aaea2b729992a3b88f3f4d707e0cbcbe35f1b82a2533db7a5',\n",
            "}\n",
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
            "# [2] consensus eval variant — embedded copy of repo scratch/run_fold_oof_eval_consensus.py (§2.98 派生)\n",
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
            "MIRROR_DIR = Path('/content/drive/MyDrive/solar_filament_v120/work/latest')\n",
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
            "# [eval] 6 arm — 3 detector の提案クラスタから合意数で放出 (§2.98)\n",
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
            "                    '--out', str(WORK / 'v120_eval.json')],\n",
            "                   capture_output=True, text=True, env=gpu_env)\n",
            "print(r.stdout[-4000:]); print('STDERR:', r.stderr[-2000:])\n",
            "if r.returncode != 0 or not (WORK / 'v120_eval.json').exists():\n",
            "    raise RuntimeError(f'eval failed rc={r.returncode}')\n",
            "mirror()\n",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "# 結果サマリ — 事前固定ゲート判定 (§2.98)\n",
            "rep = json.load(open(WORK / 'v120_eval.json'))\n",
            "arms = rep['arms']\n",
            "pq = lambda a: arms[a]['pq_by_threshold']['0.45']\n",
            "print(json.dumps({a: r['pq_by_threshold'] for a, r in arms.items()}, indent=2))\n",
            "print('emitted/image:', rep.get('emitted_per_arm'))\n",
            "base = pq('baseline::prod')\n",
            "deltas = {a: round(pq(a) - base, 6) for a in arms if a != 'baseline::prod'}\n",
            "best_arm, best_delta = max(deltas.items(), key=lambda kv: kv[1])\n",
            "gate = {\n",
            "  'prod_pq_0.45': base,\n",
            "  'sanity_prod_near_v76': abs(base - 0.444238) < 0.001,\n",
            "  'prod_diff_vs_v76': round(base - 0.444238, 6),\n",
            "  'delta_by_arm': deltas,\n",
            "  'best_arm': best_arm, 'best_delta': best_delta,\n",
            "  'decision': ('promote_candidate' if best_delta >= 0.010\n",
            "               else 'reject' if best_delta < 0 else 'reject_noise'),\n",
            "}\n",
            "print(json.dumps(gate, indent=2))\n",
            "Path(WORK / 'v120_summary.json').write_text(json.dumps(\n",
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
           / "solar_filament_v120_consensus_colab.ipynb")
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()
