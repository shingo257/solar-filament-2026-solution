"""Implementation helpers for `v68_magnetogram_probe` — extracted verbatim from `v68_magnetogram_probe.py`.

Holds the helper functions and module constants so `v68_magnetogram_probe.py` stays a readable
orchestrator. Loaded by the wrapper via sibling import.
"""


from __future__ import annotations


import pickle
import re
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
import cv2
import numpy as np
from astropy.io import fits
from pycocotools import mask as mask_utils
from scipy.ndimage import (
    gaussian_filter,
)


LOG = Path("output/v68_probe.log")


def log(*a):
    msg = " ".join(str(x) for x in a)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as h:
        h.write(time.strftime("%H:%M:%S") + " " + msg + "\n")
    print(msg, flush=True)


DRIVE = Path("G:/マイドライブ/solar_filament_v30")


CKPT = DRIVE / "artifacts" / "v64_ckpt_fold2"


PREPROC = DRIVE / "preprocessed_v30_q96"


MAG_DIR = Path("output/magnetograms")


MANIFEST = Path("output/v68_magnetogram_probe_manifest.json")


SITE2 = {"B": "bb", "C": "ct", "L": "le", "M": "ml", "T": "td", "U": "ud"}


BASE = "https://gong.nso.edu/data/magmap/QR/bqr"


NAME_RE = re.compile(r"(\d{14})([A-Za-z])h")


IMG = 2048


FEATURE_KEYS = ["conf", "probmass", "ref_raw_iou", "raw_area", "ref_area",
                "area_ratio", "aspect", "solidity", "darkness", "limb"]


MAG_KEYS = ["mag_mean", "mag_absmean", "mag_grad", "pil_dist", "pil_cover"]


CONF_GRID = (0.05, 0.08, 0.12, 0.20, 0.26, 0.35)


PKEEP_GRID = (0.0, 0.3, 0.5, 0.7, 0.85, 0.93)


SPLIT_SEED = 3407


BASELINE_PQ = 0.4432


def parse_obs(file_name: str):
    m = NAME_RE.search(file_name)
    return (datetime.strptime(m.group(1), "%Y%m%d%H%M%S"), m.group(2).upper()) if m else None


def fetch(url: str, binary=False, retries: int = 3):
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = r.read()
            return data if binary else data.decode("utf-8", "replace")
        except Exception:
            if i == retries - 1:
                return None
            time.sleep(2.0 * (i + 1))


_day_cache: dict = {}


def day_files(site2: str, day: datetime) -> list[str]:
    key = (site2, f"{day:%y%m%d}")
    if key not in _day_cache:
        url = f"{BASE}/{day:%Y%m}/{site2}bqr{day:%y%m%d}/"
        html = fetch(url)
        _day_cache[key] = re.findall(
            rf"{site2}bqr{day:%y%m%d}t(\d{{4}})\.fits", html) if html else []
    return _day_cache[key]


def nearest_mag_url(obs: datetime, site2: str):
    best = None
    for dd in (-1, 0, 1):
        day = obs + timedelta(days=dd)
        for t4 in day_files(site2, day):
            ft = day.replace(hour=int(t4[:2]), minute=int(t4[2:]), second=0)
            d = abs((ft - obs).total_seconds() / 60)
            if best is None or d < best[0]:
                best = (d, f"{BASE}/{day:%Y%m}/{site2}bqr{day:%y%m%d}/"
                            f"{site2}bqr{day:%y%m%d}t{t4}.fits", ft)
    return best  # (dt_min, url, mag_time)


_local_idx = None


def local_mag_files(site2: str, day: datetime):
    """DL済み磁気図を (site2, yymmdd) -> [(t4, path)] で索引。HTTP不要の高速パス。"""
    global _local_idx
    if _local_idx is None:
        _local_idx = {}
        for f in MAG_DIR.glob("*.fits"):
            m = re.match(r"([a-z]{2})bqr(\d{6})t(\d{4})\.fits", f.name)
            if m:
                _local_idx.setdefault((m.group(1), m.group(2)),
                                      []).append((m.group(3), f))
    return _local_idx.get((site2, f"{day:%y%m%d}"), [])


def ensure_magnetogram(file_name: str):
    parsed = parse_obs(file_name)
    if not parsed:
        return None
    obs, site = parsed
    site2 = SITE2[site]
    # まずDL済みファイルから最近傍を引く(前回DL分の再利用+サーバ負荷回避)
    best = None
    for dd in (-1, 0, 1):
        day = obs + timedelta(days=dd)
        for t4, f in local_mag_files(site2, day):
            ft = day.replace(hour=int(t4[:2]), minute=int(t4[2:]))
            d = abs((ft - obs).total_seconds() / 60)
            if best is None or d < best[0]:
                best = (d, f)
    if best and best[0] <= 60:
        return best[1]
    hit = nearest_mag_url(obs, site2)
    if not hit or hit[0] > 60:
        return None
    _, url, _ = hit
    dest = MAG_DIR / url.rsplit("/", 1)[-1]
    if not dest.exists():
        data = fetch(url, binary=True)
        if not data or len(data) < 10_000:
            return None
        dest.write_bytes(data)
    return dest


def load_mag_array(path: Path) -> np.ndarray:
    with fits.open(path) as hdul:
        data = hdul[0].data
        if data is None:
            data = hdul[1].data
    return np.asarray(data, dtype=np.float32)


def fit_disk(gray: np.ndarray):
    disk = gray > max(20.0, float(gray.max()) * 0.15)
    ys, xs = np.where(disk)
    return xs.mean(), ys.mean(), float(np.sqrt(disk.sum() / np.pi))


def mag_to_image_grid(mag: np.ndarray, cx: float, cy: float, radius: float):
    """bqr(disk内接・中心≡画像中心と仮定)を H-α の disk に合わせて 2048x2048 へ写像。"""
    m = mag.shape[0]
    # bqrのdisk半径: 有効領域の縁から推定(中央値ベースで外れ値に強い)
    center = (m - 1) / 2.0
    yy, xx = np.mgrid[:m, :m]
    rr = np.hypot(xx - center, yy - center)
    # 半径内(|B|の中央値が非ノイズ)の最大距離をdisk半径とみなす
    edge_band = np.abs(rr - m / 2.0) < 2
    r_mag = float(np.median(rr[edge_band])) if edge_band.any() else m / 2.0 - 2
    scale = radius / r_mag
    # 出力グリッド座標 -> 磁気図座標
    out_y, out_x = np.mgrid[:IMG, :IMG]
    src_x = (out_x - cx) / scale + center
    src_y = (out_y - cy) / scale + center
    mapped = cv2.remap(mag, src_x.astype(np.float32), src_y.astype(np.float32),
                        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0.0)
    return mapped


def pil_map(mag2048: np.ndarray, sigma=60.0):
    """分極反転線: 平滑化した磁場の符号が変わる場所。
    bqr(180px)を2048へ~11.4倍拡大しているので、sigmaはtarget grid基準で
    ソース数px相当(≈60px)を指定しないと微細ノイズの符号反転を拾いすぎる。"""
    sm = gaussian_filter(mag2048, sigma)
    sign = np.sign(sm)
    # 符号が隣接画素と異なる境界
    edge = np.zeros_like(sign, dtype=bool)
    edge[:-1, :] |= (sign[:-1, :] * sign[1:, :]) < 0
    edge[:, :-1] |= (sign[:, :-1] * sign[:, 1:]) < 0
    edge = cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return edge


def mag_features(mag2048, grad, dist_to_pil, mask):
    if not mask.any():
        return None
    inside = mag2048[mask]
    return {
        "mag_mean": float(np.mean(inside)),
        "mag_absmean": float(np.mean(np.abs(inside))),
        "mag_grad": float(np.mean(grad[mask])),
        "pil_dist": float(np.mean(dist_to_pil[mask])),
        "pil_cover": float(np.mean(dist_to_pil[mask] <= 120.0)),  # ~ソース10px以内の割合
    }


def decode_rle_of(p):
    rle = p.get("refined_rle") or p.get("refined")
    return mask_utils.decode(rle) != 0


def load_rows(split):
    rows = []
    for f in sorted((CKPT / split).glob("*.pkl")):
        with open(f, "rb") as h:
            rows.append(pickle.load(h))
    return rows
