"""Image I/O helpers that survive non-ASCII Windows paths."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np


def imread_unicode(path: Path) -> np.ndarray | None:
    """Read an image via ``cv2.imdecode``.

    ``cv2.imread`` fails on Windows paths containing non-ASCII characters such as
    ``G:\\マイドライブ``. Returns ``None`` when the file is missing.
    """

    if not path.exists():
        return None
    return cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)


def imread_unicode_with_retry(
    path: Path,
    attempts: int = 4,
    backoff: float = 2.0,
) -> np.ndarray:
    """Retry wrapper for flaky Drive mounts."""

    last_error: OSError | None = None
    for attempt in range(attempts):
        try:
            image = imread_unicode(path)
            if image is None:
                raise OSError(f"image missing or unreadable: {path}")
            return image
        except OSError as error:
            last_error = error
            if attempt < attempts - 1:
                time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"{path} の読み込みに{attempts}回失敗した: {last_error}")
