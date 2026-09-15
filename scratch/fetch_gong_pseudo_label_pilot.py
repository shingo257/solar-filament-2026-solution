"""pseudo-labelingパイロット用に、NSO GONGアーカイブから追加のH-alpha観測を
取得する(2018〜2022年に偏重、既存707観測との年代偏りを是正する狙い)。

## 背景

現有707観測は2011〜2017年が618件(87%)、2018〜2022年はわずか89件(13%)と
大きく偏っている。この偏りが検出器の汎化性能(特に密集シーンでの検出漏れ)を
損なっている可能性がある(README.md §2.1参照)。

## データ出典

NSO GONGアーカイブ(https://gong2.nso.edu/ftp/HA/haf/)の科学グレード
Fpack圧縮FITS(2048x2048, int16)。`hag`フォルダ(JPG)は文字ラベル・
装飾枠が焼き込まれたプレゼンテーション用画像で使えないことを確認済み
(2026-08-29)。FITSをヒストグラムマッチングで実際の学習データと同じ
階調に変換する(このスクリプトではダウンロードのみ、変換は別スクリプト)。

使い方:
    python scratch/fetch_gong_pseudo_label_pilot.py --count 150
"""

from __future__ import annotations

import argparse
import random
import re
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path

BASE = "https://gong2.nso.edu/ftp/HA/haf"
SITE_CODES = ["B", "C", "L", "M", "T", "U"]  # Big Bear/Cerro Tololo/Learmonth/Mauna Loa/Teide/Udaipur
SEED = 3407
TIMEOUT = 20
FILE_RE = re.compile(r'href="(\d{14}([A-Za-z])h\.fits\.fz)"')


def fetch_url(url: str) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
            return response.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        print(f"  警告: {url} 取得失敗 ({exc})")
        return None


def list_day_files(day: date) -> list[tuple[str, str]]:
    """(ファイル名, サイトコード) のリストを返す。"""
    url = f"{BASE}/{day:%Y%m}/{day:%Y%m%d}/"
    html = fetch_url(url)
    if not html:
        return []
    return [(m.group(1), m.group(2)) for m in FILE_RE.finditer(html)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=150)
    parser.add_argument("--years", type=int, nargs="+", default=[2018, 2019, 2020, 2021, 2022])
    parser.add_argument("--out", type=Path, default=Path("output/pseudo_label_raw/fits"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    per_year = -(-args.count // len(args.years))  # ceil

    # 各年に均等配分した候補日(月をまんべんなくカバー)を作る
    candidate_days = []
    for year in args.years:
        start = date(year, 1, 1)
        end = date(year, 12, 31)
        span = (end - start).days
        step = max(1, span // (per_year * 2))  # 候補を目標の2倍作って後で間引く
        offset = random.randint(0, step - 1)
        day = start + timedelta(days=offset)
        while day <= end:
            candidate_days.append(day)
            day += timedelta(days=step)
    random.shuffle(candidate_days)

    downloaded = []
    site_cursor = 0
    attempted_days = 0

    for day in candidate_days:
        if len(downloaded) >= args.count:
            break
        attempted_days += 1
        files = list_day_files(day)
        if not files:
            continue
        # サイトを順繰りに優先、無ければその日にある別サイトで代替
        preferred_site = SITE_CODES[site_cursor % len(SITE_CODES)]
        site_cursor += 1
        by_site: dict[str, list[str]] = {}
        for name, site in files:
            by_site.setdefault(site, []).append(name)
        chosen_site = preferred_site if preferred_site in by_site else next(iter(by_site))
        candidates = sorted(by_site[chosen_site])
        chosen_name = candidates[len(candidates) // 2]  # その日・サイトの中央時刻を選ぶ

        out_path = args.out / chosen_name
        if out_path.exists():
            continue
        url = f"{BASE}/{day:%Y%m}/{day:%Y%m%d}/{chosen_name}"
        try:
            urllib.request.urlretrieve(url, out_path)
        except Exception as exc:  # noqa: BLE001
            print(f"  警告: {url} ダウンロード失敗 ({exc})")
            continue
        downloaded.append(chosen_name)
        if len(downloaded) % 10 == 0 or len(downloaded) == args.count:
            print(f"  {len(downloaded)}/{args.count}  (候補日 {attempted_days} 件走査)", flush=True)
        time.sleep(0.3)  # サーバーへの配慮

    print(f"完了: {len(downloaded)}件を {args.out} へ保存(候補日 {attempted_days} 件走査)")
    (args.out.parent / "pilot_manifest.txt").write_text("\n".join(downloaded), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
