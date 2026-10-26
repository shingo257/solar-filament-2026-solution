"""GONG haf アーカイブから SSL 事前学習用の無ラベル gray コーパスを構築する。

背景 (RESEARCH_LOG §2.22): v70 深層 verifier は fold2 AUC 0.638 で
conf (0.797) に届かなかった。仮説の一つは「2.6k crops では視覚特徴を
学べない」で、残るレバーはドメイン事前学習。GONG アーカイブは
公開・無料 (competition rule §6a の「全員が無償でアクセス可能」を満たす)。

漏洩ガード: competition train/test の全観測から
**±exclusion_minutes (既定360分、境界を含む) 内を全棄却**。
train に含まれる verifier select/eval fold も保護するため、ラベルの使用有無に
かかわらず同じ時間窓を適用する。site を跨いでも時間近接は除外する。
旧ポリシーのコーパスは指紋不一致として resume を拒否する。

出力: gray 512x512 uint8 npz (batch 毎) + manifest.jsonl
(URL・sha256・時刻・site の完全な追跡情報)。前処理は raw gray を保存し、
solar_three_channel 変換は kernel 側で適用する (生データを凍結するため)。

使い方:
    python scratch/build_gong_ssl_corpus.py --days-per-month 4 --frames-per-day 12 \
        --out output/gong_ssl_corpus [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re

import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm

BASE = "https://gong2.nso.edu/ftp/HA/haf"
TIMEOUT = 30
NAME_RE = re.compile(r"([0-9]{14})([BCLMTUbclmtu])h\.fits\.fz")
FILE_RE = re.compile(r'''href=["']([0-9]{14}[BCLMTUbclmtu]h\.fits\.fz)["']''')
IMG_SIZE = 512
GUARD_POLICY_VERSION = "train-test-temporal-v2"


def parse_stamp(name: str) -> datetime:
    match = NAME_RE.fullmatch(name)
    if not match:
        raise ValueError(name)
    return datetime.strptime(match.group(1), "%Y%m%d%H%M%S").replace(
        tzinfo=timezone.utc)


def fetch_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "ssl-corpus/1.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as response:
        resolved = response.geturl()
        if resolved != url or response.status != 200:
            raise ValueError(f"unexpected response: {response.status} {resolved}")
        return response.read()


def list_day(day: datetime) -> list[str]:
    url = f"{BASE}/{day:%Y%m}/{day:%Y%m%d}/"
    html = fetch_bytes(url).decode("utf-8", "replace")
    names = [m for m in FILE_RE.findall(html) if NAME_RE.fullmatch(m)]
    return sorted(set(names))


def batch_frame_count(path: Path) -> int:
    """batch npz に保存された実フレーム数 (最終 batch は部分になりうる)。

    npz の key 順は保証されないため、最初の 1 次元以上の配列をフレーム
    配列とみなす (batch_size 等のスカラー補助キーを除外)。
    """
    with np.load(path) as data:
        for key in data.files:
            if data[key].ndim >= 1:
                return int(data[key].shape[0])
    raise ValueError(f"frame array not found in {path}")


def corpus_config_fingerprint(test_stamps, train_stamps, args) -> str:
    """corpus 定義入力の指紋 (漏洩ガード境界 + サンプリング規則)。

    resume 時に --test-dir/--train-json/--exclusion-minutes/年度/サンプリング
    設定が変わっていた場合、旧境界で採用したフレームが混入するのを防ぐ。
    """
    h = hashlib.sha256()
    for stamp in sorted({s.isoformat() for s in test_stamps}):
        h.update(stamp.encode())
    for stamp in sorted({s.isoformat() for s in train_stamps}):
        h.update(b"train:" + stamp.encode())
    h.update(json.dumps({
        "guard_policy": GUARD_POLICY_VERSION,
        "year": args.year, "days_per_month": args.days_per_month,
        "frames_per_day": args.frames_per_day,
        "exclusion_minutes": args.exclusion_minutes,
        "batch_size": args.batch_size, "img_size": IMG_SIZE},
        sort_keys=True).encode())
    return h.hexdigest()[:16]


def corpus_config(test_stamps, train_stamps, args) -> dict:
    """Persist the inputs needed to independently recompute the temporal guard."""
    return {"fingerprint": corpus_config_fingerprint(test_stamps, train_stamps, args),
            "guard_policy": GUARD_POLICY_VERSION, "img_size": IMG_SIZE,
            **{key: getattr(args, key) for key in (
                "year", "days_per_month", "frames_per_day", "exclusion_minutes", "batch_size")},
            "train_stamps": sorted({s.isoformat() for s in train_stamps}),
            "test_stamps": sorted({s.isoformat() for s in test_stamps})}


def prepare_output_dir(out: Path, batch_size: int, resume: bool,
                       config_fp: str | None = None):
    """既存コーパス dir を検証し、(done_files, batch_idx, kept) を返す。

    既存出力があるのに resume=False なら SystemExit (silent な上書き・
    manifest 二重 append を防ぐ)。resume=True では manifest を
    npz 保存済みフレーム境界 (各 batch の実フレーム数合計) で切り詰める —
    中断時に batch npz 未保存の行は捨てて再 DL する。batch_size が既存
    batch と不一致、または manifest が npz 収録フレーム数を下回る
    破損状態なら SystemExit。
    """
    manifest = out / "manifest.jsonl"
    batches = sorted(out.glob("batch_*.npz"))
    if batches or manifest.exists():
        if not resume:
            raise SystemExit(
                f"--out に既存コーパスがあります: {out}。"
                "継続するなら --resume、新規構築なら空の dir を指定してください")
        if batches:
            with np.load(batches[0]) as first:
                stored_bs = int(first["batch_size"]) \
                    if "batch_size" in first.files else batch_size
            if stored_bs != batch_size:
                raise SystemExit(
                    f"既存 batch_size={stored_bs} と --batch-size={batch_size} が"
                    "不一致です。同じ batch_size で --resume してください")
        if config_fp is not None:
            config_file = out / "corpus_config.json"
            if not config_file.exists():
                raise SystemExit(
                    f"corpus_config.json がありません: {out}。旧境界との"
                    "不一致を検証できないため resume できません")
            stored_fp = json.loads(
                config_file.read_text(encoding="utf-8")).get("fingerprint")
            if stored_fp != config_fp:
                raise SystemExit(
                    "corpus 定義入力 (除外境界・サンプリング規則) が既存出力と"
                    "不一致です。設定を元に戻すか別の --out を指定してください")
    if config_fp is not None:
        (out / "corpus_config.json").write_text(
            json.dumps({"fingerprint": config_fp}, indent=2),
            encoding="utf-8")
    rows = manifest.read_text(encoding="utf-8").splitlines() \
        if manifest.exists() else []
    stored = sum(batch_frame_count(b) for b in batches)
    if len(rows) < stored:
        raise SystemExit(
            f"manifest {len(rows)} 行 < batch npz 収録 {stored} フレームで"
            "不整合です (npz が manifest 未記載フレームを保持)。corpus を"
            "再構築してください")
    keep_rows = rows[:stored]
    dropped = len(rows) - len(keep_rows)
    manifest.write_text("".join(r + "\n" for r in keep_rows), encoding="utf-8")
    if dropped:
        print(f"resume: manifest を npz 保存境界へ {dropped} 行切り詰め",
              flush=True)
    done = {json.loads(r)["file"] for r in keep_rows}
    return done, len(batches), len(keep_rows)


def fits_fz_to_gray(payload: bytes) -> np.ndarray:
    """fits.fz を gray uint8 512x512 に。失敗は呼び出し側で記録して skip。"""
    import io
    from astropy.io import fits
    with fits.open(io.BytesIO(payload), memmap=False) as hdul:
        image = None
        for hdu in hdul:
            if getattr(hdu, "data", None) is not None and np.ndim(hdu.data) == 2:
                image = np.asarray(hdu.data, dtype=np.float64)
                break
    if image is None:
        raise ValueError("no 2-D image HDU")
    image = np.nan_to_num(image, nan=0.0)
    lo, hi = np.percentile(image, [0.5, 99.5])
    gray = np.clip((image - lo) / max(hi - lo, 1e-9), 0, 1)
    import cv2
    gray8 = (gray * 255).astype(np.uint8)
    return cv2.resize(gray8, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path,
                        default=Path("G:/マイドライブ/solar_filament_v30/"
                                     "competition_data/MAGFiLO_1.0_Kaggle_2026/test/test_images"))
    parser.add_argument("--train-json", type=Path,
                        default=Path("G:/マイドライブ/solar_filament_v30/"
                                     "competition_data/MAGFiLO_1.0_Kaggle_2026/train/"
                                     "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    parser.add_argument("--year", type=int, default=2011)
    parser.add_argument("--days-per-month", type=int, default=4)
    parser.add_argument("--frames-per-day", type=int, default=12)
    parser.add_argument("--exclusion-minutes", type=int, default=360)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--out", type=Path, default=Path("output/gong_ssl_corpus"))
    parser.add_argument("--dry-run", action="store_true",
                        help="listing+件数確認のみ (DL しない)")
    parser.add_argument("--resume", action="store_true",
                        help="既存の --out dir の続きから再開する")
    args = parser.parse_args()
    if args.exclusion_minutes < 0:
        raise SystemExit("--exclusion-minutes must be non-negative")
    args.out.mkdir(parents=True, exist_ok=True)

    # 除外集合: test 観測の timestamp (±window) — site 不問で時間近接を除外
    test_stamps = []
    for name in sorted(args.test_dir.iterdir()):
        m = NAME_RE.match(name.name.replace(".jpeg", ".fits.fz"))
        if m:
            test_stamps.append(parse_stamp(m.group(0)))
    if not test_stamps:
        raise SystemExit(f"test 観測が読めません: {args.test_dir}")
    # 全 train 観測を時間窓で除外し、verifier の select/eval fold も保護する。
    with open(args.train_json, encoding="utf-8") as handle:
        coco = json.load(handle)
    images = coco.get("images") if isinstance(coco, dict) else None
    if not isinstance(images, list) or not images:
        raise SystemExit("train observations must be a non-empty images list")
    try:
        train_stamps = {parse_stamp(image["file_name"].replace(".jpeg", ".fits.fz"))
                        for image in images}
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        raise SystemExit("train observation timestamp is missing or invalid") from error
    observation_stamps = set(test_stamps) | train_stamps
    print({"test_excl_anchors": len(test_stamps),
           "train_excl_anchors": len(train_stamps),
           "exclusion_min": args.exclusion_minutes,
           "guard_policy": GUARD_POLICY_VERSION})

    def near_observation(stamp: datetime) -> bool:
        return any(abs(stamp - t) <= timedelta(minutes=args.exclusion_minutes)
                   for t in observation_stamps)

    manifest = args.out / "manifest.jsonl"
    config_fp = corpus_config_fingerprint(test_stamps, train_stamps, args)
    done, batch_idx, kept = prepare_output_dir(
        args.out, args.batch_size, args.resume, config_fp=config_fp)
    (args.out / "corpus_config.json").write_text(
        json.dumps(corpus_config(test_stamps, train_stamps, args), indent=2),
        encoding="utf-8")
    if done:
        print(f"resume: {kept} フレーム取得済み・batch {batch_idx} から再開",
              flush=True)
    done_stamps = {parse_stamp(name) for name in done}
    frames: list[np.ndarray] = []
    skipped = failed = 0
    days_per_month = args.days_per_month
    for month in range(1, 13):
        # 月内に均等に日を配置
        days_in_month = 31 if month in (1, 3, 5, 7, 8, 10, 12) else \
            (29 if month == 2 and args.year % 4 == 0 else
             28 if month == 2 else 30)
        offsets = sorted({round(1 + i * (days_in_month - 1) / max(days_per_month - 1, 1))
                          for i in range(days_per_month)})
        for dom in offsets:
            day = datetime(args.year, month, dom, tzinfo=timezone.utc)
            try:
                names = list_day(day)
            except Exception as error:  # noqa: BLE001
                print(f"{day:%Y-%m-%d} listing failed: {error}", flush=True)
                continue
            # sparse sampling: 時間帯を均等に
            eligible = [n for n in names]
            if not eligible:
                continue
            step = max(len(eligible) // args.frames_per_day, 1)
            picked = eligible[::step][:args.frames_per_day]
            for name in picked:
                stamp = parse_stamp(name)
                if near_observation(stamp):
                    skipped += 1
                    continue
                if name in done:
                    continue  # resume: 既に batch npz + manifest に収録済み
                if stamp in done_stamps:
                    continue
                url = f"{BASE}/{day:%Y%m}/{day:%Y%m%d}/{name}"
                if args.dry_run:
                    kept += 1
                    continue
                try:
                    payload = fetch_bytes(url)
                    gray = fits_fz_to_gray(payload)
                except Exception as error:  # noqa: BLE001
                    failed += 1
                    tqdm.write(f"DL/decode失敗 {name}: {error}")
                    continue
                frames.append(gray)
                done_stamps.add(stamp)
                kept += 1
                with manifest.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({
                        "file": name, "url": url,
                        "utc": stamp.isoformat(),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "frame_sha256": hashlib.sha256(gray.tobytes()).hexdigest(),
                        "bytes": len(payload)}) + "\n")
                if len(frames) >= args.batch_size:
                    np.savez_compressed(
                        args.out / f"batch_{batch_idx:03d}.npz",
                        np.stack(frames), batch_size=args.batch_size)
                    frames, batch_idx = [], batch_idx + 1
                    tqdm.write(f"batch_{batch_idx - 1:03d} saved ({kept} frames)")
                time.sleep(0.2)  # archive への配慮
        print(f"month {month:02d}: kept={kept} skipped={skipped} "
              f"failed={failed}", flush=True)
    if frames:
        np.savez_compressed(args.out / f"batch_{batch_idx:03d}.npz",
                            np.stack(frames), batch_size=args.batch_size)
    summary = {"kept": kept, "skipped_observation_window": skipped, "failed": failed,
               "img_size": IMG_SIZE, "dry_run": args.dry_run,
               "exclusion_minutes": args.exclusion_minutes,
               "guard_policy": GUARD_POLICY_VERSION}
    (args.out / "corpus_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
