from __future__ import annotations

import random

import numpy as np
from numpy.typing import NDArray

from rgbt_uav.data.manifest import synchronized_seed


def synchronized_horizontal_flip(
    rgb: NDArray[np.generic],
    ir: NDArray[np.generic],
    boxes: list[list[float]],
    *,
    pair_id: str,
    epoch: int,
    global_seed: int,
    probability: float = 0.5,
) -> tuple[NDArray[np.generic], NDArray[np.generic], list[list[float]]]:
    """Apply one deterministic geometric decision to an RGB/IR pair and its HBBs."""

    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0, 1]")
    if rgb.ndim not in (2, 3) or ir.ndim not in (2, 3):
        raise ValueError("modalities must be HxW or HxWxC arrays")
    if rgb.shape[:2] != ir.shape[:2]:
        raise ValueError("RGB and IR spatial shapes must match")

    copied_boxes = [[float(value) for value in box] for box in boxes]
    generator = random.Random(
        synchronized_seed(pair_id, epoch=epoch, global_seed=global_seed)
    )
    if generator.random() >= probability:
        return rgb.copy(), ir.copy(), copied_boxes

    width = float(rgb.shape[1])
    flipped_boxes = [
        [width - box[2], box[1], width - box[0], box[3]] for box in copied_boxes
    ]
    return np.flip(rgb, axis=1).copy(), np.flip(ir, axis=1).copy(), flipped_boxes
