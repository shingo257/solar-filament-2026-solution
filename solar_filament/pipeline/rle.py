"""Official compressed COCO RLE codec (pure Python, pycocotools-compatible).

Re-exports the V32 offline codec so callers do not depend on the overlap-cluster
repair helpers when they only need encode/decode.
"""

from __future__ import annotations

from solar_filament.pipeline.v32_overlap_cluster import (
    HEIGHT,
    WIDTH,
    decode_compressed_counts,
    decode_mask,
    encode_compressed_counts,
    encode_mask,
)

__all__ = [
    "HEIGHT",
    "WIDTH",
    "decode_compressed_counts",
    "decode_mask",
    "encode_compressed_counts",
    "encode_mask",
]
