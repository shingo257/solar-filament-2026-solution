"""YOLO OOF packed-bitmask cache I/O."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np


def load_yolo_cache(path: Path) -> tuple[list[np.ndarray], np.ndarray]:
    """Load packed YOLO OOF masks and confidences from an ``.npz`` file.

    Empty caches return ``([], zeros(0))``. Masks are boolean arrays truncated to
    the stored width.
    """

    data = np.load(path)
    _, width = map(int, data["shape"])
    packed = data["packed"]
    if packed.size == 0:
        return [], np.zeros(0, np.float32)
    masks = [np.unpackbits(item, axis=-1)[:, :width].astype(bool) for item in packed]
    return masks, data["confidences"].astype(np.float32)


def load_yolo_cache_with_retry(
    path: Path,
    attempts: int = 4,
    backoff: float = 2.0,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Retry wrapper for flaky Google Drive network mounts (OSError errno 22)."""

    last_error: OSError | None = None
    for attempt in range(attempts):
        try:
            return load_yolo_cache(path)
        except OSError as error:
            last_error = error
            if attempt < attempts - 1:
                time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"{path} の読み込みに{attempts}回失敗した: {last_error}")
