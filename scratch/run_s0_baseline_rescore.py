"""S0-3: V35/raw YOLO/V34/V42を同じ科学endpointで再評価する。

既存のリークなしOOF cacheとfold別refinerだけを使うCPU評価。科学指標は
``--eval-size``（既定512）へ統一して計算し、公式2048 PQと混同しない。
V42は採用候補ではなくRecall-FP trade-offを測るnegative controlである。
"""


from __future__ import annotations


import sys
from pathlib import Path


try:
    from .run_s0_baseline_rescore_impl import (
        SCIENCE_ENDPOINTS,
        SEED8407,
        annotation_mask,
        evaluate_variant,
        filter_native,
        load_prediction_cache,
        prediction_cache_path,
        preferred_image_root,
        refiner_checkpoint_path,
        resize_mask,
        resolve_preprocessed_image_path,
        resolve_yolo_cache_path,
        save_prediction_cache,
        summarize_group,
    )
    from .run_s0_baseline_rescore_main import main
except ImportError:  # Direct `python scratch/...` or spec_from_file_location loading.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_s0_baseline_rescore_impl import (
        SCIENCE_ENDPOINTS,
        SEED8407,
        annotation_mask,
        evaluate_variant,
        filter_native,
        load_prediction_cache,
        prediction_cache_path,
        preferred_image_root,
        refiner_checkpoint_path,
        resize_mask,
        resolve_preprocessed_image_path,
        resolve_yolo_cache_path,
        save_prediction_cache,
        summarize_group,
    )
    from run_s0_baseline_rescore_main import main



__all__ = [
    "SCIENCE_ENDPOINTS",
    "SEED8407",
    "annotation_mask",
    "evaluate_variant",
    "filter_native",
    "load_prediction_cache",
    "prediction_cache_path",
    "preferred_image_root",
    "refiner_checkpoint_path",
    "resize_mask",
    "resolve_preprocessed_image_path",
    "resolve_yolo_cache_path",
    "save_prediction_cache",
    "summarize_group",
]


if __name__ == "__main__":
    raise SystemExit(main())
