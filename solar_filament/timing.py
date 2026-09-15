"""段階別の実行時間を測って manifest へ残すための小さなユーティリティ。

なぜ必要か:
  本コンペの定量評価（全体の70%）には「エンドツーエンドの計算効率」が含まれる。
  にもかかわらず、V19〜V35 を通じて推論時間を計測した記録が一度もない。
  技術報告書にも記載が要る。

設計方針:
  - 平均だけでなく分布（p50 / p95 / max）を出す。フィラメント数は画像ごとに大きく違い、
    平均だけでは「重い画像が何秒かかるか」が消える
  - GPU を使う段階は同期してから測る。非同期実行のまま測ると実際より速く見える
  - manifest へそのまま入る dict を返す（HANDOVER §34）

使い方:

    timer = StageTimer(device="T4")
    for image in images:
        with timer.stage("preprocess"):
            ...
        with timer.stage("detect"):
            ...
    timer.set_image_count(len(images))
    manifest["timing"] = timer.summary()
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager


def _percentile(values: list[float], fraction: float) -> float:
    """小さな標本でも破綻しない最近傍パーセンタイル。numpy 非依存。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


class StageTimer:
    """段階ごとの経過時間を貯める。

    同じ段階名で複数回計測してよい（1画像1回でも、1画像に複数回でもよい）。
    `synchronize` に torch.cuda.synchronize を渡すと GPU 段階を正しく測れる。
    """

    def __init__(self, device: str = "unknown", synchronize=None):
        self.device = device
        self._synchronize = synchronize
        self._samples: dict[str, list[float]] = defaultdict(list)
        self._image_count = 0
        self._wall_started = time.perf_counter()

    @contextmanager
    def stage(self, name: str):
        if self._synchronize is not None:
            self._synchronize()
        started = time.perf_counter()
        try:
            yield
        finally:
            if self._synchronize is not None:
                self._synchronize()
            self._samples[name].append(time.perf_counter() - started)

    def record(self, name: str, seconds: float) -> None:
        """外部で測った値を直接入れる（既存コードへ後付けする場合）。"""
        self._samples[name].append(float(seconds))

    def set_image_count(self, count: int) -> None:
        self._image_count = int(count)

    @property
    def stages(self) -> list[str]:
        return list(self._samples)

    def total(self, name: str | None = None) -> float:
        if name is not None:
            return sum(self._samples.get(name, []))
        return sum(sum(values) for values in self._samples.values())

    def summary(self) -> dict:
        """manifest へそのまま入れられる形。秒単位、小数4桁。"""
        stages = {}
        for name, values in self._samples.items():
            stages[name] = {
                "calls": len(values),
                "total_s": round(sum(values), 4),
                "mean_s": round(sum(values) / len(values), 4),
                "p50_s": round(_percentile(values, 0.50), 4),
                "p95_s": round(_percentile(values, 0.95), 4),
                "max_s": round(max(values), 4),
            }
        measured = self.total()
        wall = time.perf_counter() - self._wall_started
        report = {
            "device": self.device,
            "images": self._image_count,
            "measured_total_s": round(measured, 4),
            "wall_clock_s": round(wall, 4),
            # 計測区間の外にどれだけ時間が漏れているか。大きいなら計測が不完全
            "unmeasured_s": round(max(wall - measured, 0.0), 4),
            "stages": stages,
        }
        if self._image_count:
            report["seconds_per_image"] = round(measured / self._image_count, 4)
            report["stage_seconds_per_image"] = {
                name: round(values["total_s"] / self._image_count, 4)
                for name, values in stages.items()
            }
        return report

    def format_table(self) -> str:
        """人間が読む用。報告書へ貼れる整形済みテキスト。"""
        summary = self.summary()
        lines = [f"device={summary['device']}  images={summary['images']}  "
                 f"measured={summary['measured_total_s']:.1f}s  "
                 f"unmeasured={summary['unmeasured_s']:.1f}s",
                 f"{'stage':<16}{'calls':>7}{'total_s':>10}{'mean_s':>9}"
                 f"{'p95_s':>9}{'max_s':>9}{'s/image':>10}"]
        per_image = summary.get("stage_seconds_per_image", {})
        for name, values in sorted(summary["stages"].items(),
                                   key=lambda item: -item[1]["total_s"]):
            lines.append(
                f"{name:<16}{values['calls']:>7}{values['total_s']:>10.2f}"
                f"{values['mean_s']:>9.3f}{values['p95_s']:>9.3f}"
                f"{values['max_s']:>9.3f}{per_image.get(name, 0):>10.3f}")
        if summary.get("seconds_per_image"):
            lines.append(f"{'TOTAL':<16}{'':>7}{summary['measured_total_s']:>10.2f}"
                         f"{'':>9}{'':>9}{'':>9}{summary['seconds_per_image']:>10.3f}")
        return "\n".join(lines)
