"""StageTimer のテスト。

計測器そのものが間違っていると、報告する計算効率が嘘になる。特に
「計測区間の外に漏れた時間」を検出できることを固定しておく。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.timing import StageTimer, _percentile  # noqa: E402


def test_percentile_handles_small_samples():
    assert _percentile([], 0.5) == 0.0
    assert _percentile([1.0], 0.95) == 1.0
    assert _percentile([1.0, 2.0, 3.0], 0.0) == 1.0
    assert _percentile([1.0, 2.0, 3.0], 1.0) == 3.0
    assert _percentile([3.0, 1.0, 2.0], 0.5) == 2.0   # 入力順に依存しない


def test_stage_accumulates_per_name():
    timer = StageTimer(device="test")
    for _ in range(3):
        with timer.stage("detect"):
            time.sleep(0.01)
    with timer.stage("refine"):
        time.sleep(0.01)
    summary = timer.summary()
    assert summary["stages"]["detect"]["calls"] == 3
    assert summary["stages"]["refine"]["calls"] == 1
    assert summary["stages"]["detect"]["total_s"] >= 0.025
    assert set(summary["stages"]) == {"detect", "refine"}


def test_seconds_per_image_requires_image_count():
    timer = StageTimer()
    with timer.stage("a"):
        time.sleep(0.01)
    assert "seconds_per_image" not in timer.summary(), "画像数未設定なのに per-image が出ている"

    timer.set_image_count(4)
    summary = timer.summary()
    # 出力は小数4桁へ丸めてあるので、丸め前の値と突き合わせる
    assert summary["seconds_per_image"] == round(timer.total() / 4, 4), summary
    assert "a" in summary["stage_seconds_per_image"]


def test_unmeasured_time_is_surfaced():
    """計測区間の外で時間を使ったら、それが見えること。

    これが 0 に見えると「全部測れている」と誤解する。
    """
    timer = StageTimer()
    time.sleep(0.05)                      # 区間の外
    with timer.stage("a"):
        time.sleep(0.01)
    summary = timer.summary()
    assert summary["unmeasured_s"] >= 0.03, summary
    assert summary["measured_total_s"] < summary["wall_clock_s"]


def test_exception_inside_stage_still_records():
    """例外で抜けても計測が失われないこと（finally で記録している）。"""
    timer = StageTimer()
    try:
        with timer.stage("boom"):
            time.sleep(0.01)
            raise ValueError("expected")
    except ValueError:
        pass
    assert timer.summary()["stages"]["boom"]["calls"] == 1


def test_synchronize_hook_is_called_around_stage():
    """GPU 段階で synchronize を呼ばないと実際より速く見える。前後で呼ぶこと。"""
    calls = []
    timer = StageTimer(device="fake-gpu", synchronize=lambda: calls.append(1))
    with timer.stage("gpu"):
        pass
    assert len(calls) == 2, calls


def test_record_accepts_external_measurements():
    timer = StageTimer()
    timer.record("external", 1.5)
    timer.record("external", 0.5)
    stage = timer.summary()["stages"]["external"]
    assert stage["calls"] == 2 and abs(stage["total_s"] - 2.0) < 1e-9
    assert abs(stage["mean_s"] - 1.0) < 1e-9


def test_format_table_is_printable():
    timer = StageTimer(device="T4")
    with timer.stage("detect"):
        time.sleep(0.01)
    timer.set_image_count(2)
    text = timer.format_table()
    assert "detect" in text and "T4" in text and "TOTAL" in text
    print("\n" + text)


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print(f"PASS {name}")
            except AssertionError as error:
                failures += 1
                print(f"FAIL {name}: {error}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
