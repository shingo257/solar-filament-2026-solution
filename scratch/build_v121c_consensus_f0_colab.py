"""v121c — cons2of3_c26 の fold-0 honest 複製 eval (Colab、§2.99)。

v120 `cons2of3_c26` (Δ@0.45=+0.01155、§2.98 ゲート +0.010 通過) の
fold-0 複製 Phase B。v121a (crop folds(1,2))・v121b (yolo26m folds(1,2))
で学習した fold-0 honest detector と prod yolo_fold_0・edgeatt refiner
fold0_resume で、fold-2 と全く同じ 6 arm consensus 評価を fold-0 OOF に
対して実行する。

detector 3 本 (いずれも folds(1,2) 学習 → fold-0 OOF honest):
- `yolo`: prod yolo11m fold_0 (manifest role `yolo` binding、fold=0)
- `crop`: v121a croptrain fold0 honest 版 (fit_files のみ、trusted-pin
  = v121a manifest 記録 `datasets.croptrain.weights_sha256`)
- `yolo26`: v121b yolo26m fold0 (trusted-pin = v121b manifest 記録
  `results.yolo26m.weights_sha256`)

trusted-pin は「学習 manifest に記録された sha」を実行時に読み取り、
eval script の --trusted-sha256 へ渡す — 学習 manifest が byte
binding の正本 (§2.99)。

棄却/採用ゲート (事前固定, RESEARCH_LOG §2.99):
  - cons2of3_c26 Δ@0.45 (vs fold-0 prod arm) >= +0.005
    → `cross_fold_verified` → 採用検討へ (all-data det 学習 + LB 検証)
    未満 → `unverified_fold0` で consensus 系閉鎖
  - sanity: prod arm PQ@0.45 が fold-0 prod 確定値 0.444137 ±0.001
    (v108 §2.86)

入力 (reviewed セット + v121a/v121b 学習済み ckpt + 各 manifest):
  - Drive solar_filament_v30/: train json, test_images, preproc, folds CSV
    (sha febb3f67...), fold0 refiner ckpt, yolo_fold_0 best.pt,
    runs/v121_croptrain_f0_honest/ (漏洩修正版、旧 v121_croptrain_f0 は
    fold-0 混入のため不使用), runs/v121_yolo26_f0/,
    artifacts/v121a_*・v121b_*_colab_manifest.json
  - shingo257/solar-v73-code       : shared helpers (eval script は embedded)

provenance manifest は fold 別 (manifest["fold"] == 評価 fold が必須で、
checkpoints role は fold-2 学習済み ckpt の sha を bind する) のため、
fold-2 用 v73_reviewed_provenance.json は fold-0 では使えない。本ノートは
fold-0 ckpt の独立検証済み sha (§2.99 記録) から fold-0 manifest を
決定論的に生成し、その自己 sha を V73_FOLD0_PROVENANCE_SHA256 で照合する。
"""

from __future__ import annotations

import json
from pathlib import Path

CONSENSUS_SCRIPT_PATH = (Path(__file__).resolve().parent
                         / "run_fold_oof_eval_consensus.py")

MD_CELL = [
    "# v121c — cons2of3_c26 fold-0 honest 複製 eval (Colab)\n",
    "\n",
    "v120 で `cons2of3_c26` が Δ@0.45=+0.01155 で §2.98 ゲート通過 →\n",
    "§2.99 規定の fold-0 複製。detector 3 本 (prod yolo_fold_0・v121a\n",
    "crop folds(1,2)・v121b yolo26 folds(1,2)) の提案を fold-2 と同じ\n",
    "mask-IoU>=0.5 クラスタ+合意放出で fold-0 OOF に適用:\n",
    "\n",
    "| arm | 放出ルール |\n",
    "|---|---|\n",
    "| `prod` | yolo 提案のみ conf>=0.26 (sanity ≈0.444137) |\n",
    "| `union3` | 全クラスタ (union 参照) |\n",
    "| `union3_c26` | max conf>=0.26 のクラスタ |\n",
    "| `cons2of3` | >=2 detector 支持 (低conf合意救済) |\n",
    "| `cons2of3_c26` | conf>=0.26 の支持 >=2 (昇格候補 arm) |\n",
    "| `cons3of3` | 全員支持 |\n",
    "\n",
    "**事前固定ゲート** (RESEARCH_LOG §2.99):\n",
    "- cons2of3_c26 Δ@0.45 ≥ +0.005 → `cross_fold_verified` (採用検討へ)\n",
    "- 未満 → `unverified_fold0` で consensus 系閉鎖\n",
    "- sanity: prod arm ≈ fold-0 prod 確定値 0.444137 ±0.001\n",
    "\n",
    "**Provenance pin:** `V73_FOLD0_PROVENANCE_SHA256` (fold-0 manifest 自己 sha、\n",
    "§2.99 記録値 = `da3d53ab...`)。fold-2 manifest は fold/binding 不一致で\n",
    "使えないため、fold-0 ckpt の独立 pin からセル内生成して照合する。\n",
    "yolo det は manifest binding、crop/yolo26 det は v121a/v121b 学習\n",
    "manifest 記録の weights_sha256 を trusted-pin として受理。\n",
    "Drive `solar_filament_v35/kaggle.json` で kaggle 認証 (shingo257)。\n",
    "eval 終了時に Drive `solar_filament_v121/work/latest/` へ sha256 検証付きミラー。\n",
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
            "EXPECTED_PROVENANCE_SHA = os.environ.get('V73_FOLD0_PROVENANCE_SHA256', '')\n",
            "if not EXPECTED_PROVENANCE_SHA:\n",
            "    try:\n",
            "        from google.colab import userdata\n",
            "        EXPECTED_PROVENANCE_SHA = userdata.get('V73_FOLD0_PROVENANCE_SHA256') or ''\n",
            "    except Exception:\n",
            "        pass\n",
            "if not EXPECTED_PROVENANCE_SHA:\n",
            "    try:\n",
            "        from kaggle_secrets import UserSecretsClient\n",
            "        EXPECTED_PROVENANCE_SHA = UserSecretsClient().get_secret('V73_FOLD0_PROVENANCE_SHA256')\n",
            "    except Exception:\n",
            "        pass\n",
            "if not re.fullmatch(r'[0-9a-f]{64}', EXPECTED_PROVENANCE_SHA):\n",
            "    try:\n",
            "        EXPECTED_PROVENANCE_SHA = input('V73_FOLD0_PROVENANCE_SHA256 (fold-0 reviewed manifest hash): ').strip()\n",
            "    except (EOFError, OSError):\n",
            "        pass\n",
            "if not re.fullmatch(r'[0-9a-f]{64}', EXPECTED_PROVENANCE_SHA):\n",
            "    raise RuntimeError('Set V73_FOLD0_PROVENANCE_SHA256 after independent provenance review (§2.99: da3d53ab...)')\n",
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
            "    'baseline_ckpt': V30 / 'artifacts/mediseg_edgeattnet_refiner_fold0_resume.pt',\n",
            "    'yolo_ckpt': V30 / 'runs/yolo_fold_0/weights/best.pt',\n",
            "    'crop_ckpt': V30 / 'runs/v121_croptrain_f0_honest/yolo_croptrain_fold0/weights/best.pt',\n",
            "    'yolo26_ckpt': V30 / 'runs/v121_yolo26_f0/yolo_yolo26m_fold0/weights/best.pt',\n",
            "    'v121a_manifest': V30 / 'artifacts/v121a_croptrain_f0_colab_manifest.json',\n",
            "    'v121b_manifest': V30 / 'artifacts/v121b_yolo26_f0_colab_manifest.json',\n",
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
            "# §2.99 — crop/yolo26 fold-0 重みの trusted-pin は各学習 manifest が正本\n",
            "# (学習時に記録された weights_sha256 を読み取り、eval cell で byte 照合)\n",
            "_v121a_man = json.loads(inputs['v121a_manifest'].read_text(encoding='utf-8'))\n",
            "_v121b_man = json.loads(inputs['v121b_manifest'].read_text(encoding='utf-8'))\n",
            "TRUSTED_SHA256 = {\n",
            "    str(crop_ckpt): _v121a_man['datasets']['croptrain']['weights_sha256'],\n",
            "    str(yolo26_ckpt): _v121b_man['results']['yolo26m']['weights_sha256'],\n",
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
            "\n",
            "code_dir = next(p.parent for p in IN.rglob('train_refiner_fold.py'))\n",
            "sys.path.insert(0, str(code_dir))\n",
            "from v73_split_provenance import (load_reviewed_manifest, load_fold_mapping,\n",
            "                                  validate_checkpoint_binding)\n",
            "# §2.99 — fold-0 reviewed manifest は fold-0 ckpt の独立 pin から\n",
            "# 決定論的に生成し、自己 sha を外部 pin (V73_FOLD0_PROVENANCE_SHA256)\n",
            "# で照合する。fold-2 manifest は fold/binding が fold-0 と不一致のため\n",
            "# 使用しない (checkpoints.yolo/baseline は fold-2 重みの sha)。\n",
            "FOLD0_YOLO_SHA = 'ab8810916e34a755cc0d7a705dbb7ade60338ae247f8baac02adfe64026c0d35'\n",
            "FOLD0_REFINER_SHA = 'caa8b27bbdaa888019a599d001b2d4dee844dbfee90186a18f8e0db95ce8b264'\n",
            "fold0_manifest = {'schema_version': 1, 'fold': 0,\n",
            "                  'folds_sha256': EXPECTED_FOLDS_SHA,\n",
            "                  'checkpoints': {'yolo': FOLD0_YOLO_SHA,\n",
            "                                  'baseline': FOLD0_REFINER_SHA}}\n",
            "manifest_path = WORK / 'v73_reviewed_provenance_fold0.json'\n",
            "manifest_path.write_text(json.dumps(fold0_manifest, sort_keys=True,\n",
            "                                    separators=(',', ':')), encoding='utf-8')\n",
            "manifest = load_reviewed_manifest(manifest_path, EXPECTED_PROVENANCE_SHA, 0)\n",
            "if manifest['folds_sha256'] != EXPECTED_FOLDS_SHA:\n",
            "    raise RuntimeError('reviewed split differs')\n",
            "coco = json.loads(train_json.read_text())\n",
            "load_fold_mapping(folds_csv, {i['file_name'] for i in coco['images']}, 0,\n",
            "                  manifest['folds_sha256'])\n",
            "validate_checkpoint_binding(yolo_ckpt, 'yolo', manifest)\n",
            "validate_checkpoint_binding(baseline_ckpt, 'baseline', manifest)\n",
            "\n",
            "# [2] consensus eval variant — embedded copy of repo scratch/run_fold_oof_eval_consensus.py (§2.98 派生・fold-0 実行)\n",
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
            "MIRROR_DIR = Path('/content/drive/MyDrive/solar_filament_v121/work/latest')\n",
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
            "# [eval] 6 arm — fold-0 OOF で 3 detector の提案クラスタから合意数で放出 (§2.99)\n",
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
            "                    '--folds-csv', str(folds_csv), '--fold', '0',\n",
            "                    '--provenance-manifest', str(manifest_path),\n",
            "                    '--provenance-sha256', EXPECTED_PROVENANCE_SHA,\n",
            "                    '--detector', f'yolo={yolo_ckpt}',\n",
            "                    '--detector', f'crop={crop_ckpt}',\n",
            "                    '--detector', f'yolo26={yolo26_ckpt}',\n",
            "                    '--model', 'edgeatt',\n",
            "                    *trusted_args,\n",
            "                    '--refiner', f'baseline={baseline_ckpt}',\n",
            "                    '--out', str(WORK / 'v121_eval.json')],\n",
            "                   capture_output=True, text=True, env=gpu_env)\n",
            "print(r.stdout[-4000:]); print('STDERR:', r.stderr[-2000:])\n",
            "if r.returncode != 0 or not (WORK / 'v121_eval.json').exists():\n",
            "    raise RuntimeError(f'eval failed rc={r.returncode}')\n",
            "mirror()\n",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "# 結果サマリ — 事前固定ゲート判定 (§2.99: cons2of3_c26 の fold-0 複製)\n",
            "rep = json.load(open(WORK / 'v121_eval.json'))\n",
            "arms = rep['arms']\n",
            "pq = lambda a: arms[a]['pq_by_threshold']['0.45']\n",
            "print(json.dumps({a: r['pq_by_threshold'] for a, r in arms.items()}, indent=2))\n",
            "print('emitted/image:', rep.get('emitted_per_arm'))\n",
            "base = pq('baseline::prod')\n",
            "deltas = {a: round(pq(a) - base, 6) for a in arms if a != 'baseline::prod'}\n",
            "rep_delta = deltas['baseline::cons2of3_c26']  # §2.99 の複製対象 arm\n",
            "gate = {\n",
            "  'prod_pq_0.45': base,\n",
            "  'sanity_prod_near_v108': abs(base - 0.444137) < 0.001,\n",
            "  'prod_diff_vs_v108': round(base - 0.444137, 6),\n",
            "  'delta_by_arm': deltas,\n",
            "  'cons2of3_c26_delta_fold0': rep_delta,\n",
            "  'fold2_reference_delta': 0.01155,\n",
            "  'decision': ('cross_fold_verified' if rep_delta >= 0.005\n",
            "               else 'unverified_fold0'),\n",
            "}\n",
            "print(json.dumps(gate, indent=2))\n",
            "Path(WORK / 'v121_summary.json').write_text(json.dumps(\n",
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
           / "solar_filament_v121c_consensus_f0_colab.ipynb")
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()
