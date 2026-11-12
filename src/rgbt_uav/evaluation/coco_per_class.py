"""Per-class metric extraction from a COCOeval precision tensor."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _mean_valid(
    values: np.ndarray,
) -> float | None:
    valid = values[
        np.isfinite(values) & (values > -1)
    ]
    if valid.size == 0:
        return None
    return float(valid.mean())


def _iou_index(
    thresholds: np.ndarray,
    target: float,
) -> int:
    matches = np.flatnonzero(
        np.isclose(
            thresholds,
            target,
            rtol=0.0,
            atol=1e-8,
        )
    )
    if matches.size != 1:
        raise ValueError(
            f"COCO IoU threshold {target:.2f} is missing"
        )
    return int(matches[0])


def summarize_coco_per_class(
    precision: np.ndarray,
    *,
    iou_thresholds: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, dict[str, float | None]]:
    """Extract per-class AP50-95, AP50, AP75, and AP-small."""

    if not isinstance(precision, np.ndarray):
        raise TypeError(
            "precision must be a numpy array"
        )
    if precision.ndim != 5:
        raise ValueError(
            "precision must have five dimensions"
        )
    if precision.shape[2] != len(class_names):
        raise ValueError(
            "precision class dimension does not match class_names"
        )
    if precision.shape[3] < 2:
        raise ValueError(
            "precision must contain all and small area ranges"
        )
    if precision.shape[4] < 1:
        raise ValueError(
            "precision must contain max-det results"
        )

    thresholds = np.asarray(
        iou_thresholds,
        dtype=np.float64,
    )
    if thresholds.ndim != 1:
        raise ValueError(
            "iou_thresholds must be one-dimensional"
        )
    if precision.shape[0] != thresholds.shape[0]:
        raise ValueError(
            "precision IoU dimension does not match thresholds"
        )

    ap50_index = _iou_index(thresholds, 0.50)
    ap75_index = _iou_index(thresholds, 0.75)
    max_det_index = precision.shape[4] - 1

    result: dict[
        str,
        dict[str, float | None],
    ] = {}

    for class_index, class_name in enumerate(
        class_names
    ):
        if not isinstance(class_name, str) or not class_name:
            raise ValueError(
                "class names must be non-empty strings"
            )

        all_iou = precision[
            :,
            :,
            class_index,
            0,
            max_det_index,
        ]
        at_50 = precision[
            ap50_index,
            :,
            class_index,
            0,
            max_det_index,
        ]
        at_75 = precision[
            ap75_index,
            :,
            class_index,
            0,
            max_det_index,
        ]
        small = precision[
            :,
            :,
            class_index,
            1,
            max_det_index,
        ]

        result[class_name] = {
            "AP50-95": _mean_valid(all_iou),
            "AP50": _mean_valid(at_50),
            "AP75": _mean_valid(at_75),
            "AP_small": _mean_valid(small),
        }

    return result