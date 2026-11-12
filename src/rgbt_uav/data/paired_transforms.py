from __future__ import annotations

import hashlib
import math

import torch
from torch import Tensor


class PairedTransformError(ValueError):
    """Raised when a synchronized RGB-IR transform receives invalid input."""


def _validate_inputs(rgb: Tensor, ir: Tensor, boxes: Tensor) -> None:
    if rgb.ndim != 3 or ir.ndim != 3:
        raise PairedTransformError("RGB and IR images must be CHW tensors")

    if rgb.shape[1:] != ir.shape[1:]:
        raise PairedTransformError(
            "RGB and IR images must have the same spatial shape"
        )

    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise PairedTransformError(
            "boxes must have shape [num_boxes, 4]"
        )

    if not torch.is_floating_point(boxes):
        raise PairedTransformError("boxes must be floating-point values")

    if not torch.isfinite(boxes).all():
        raise PairedTransformError("boxes must contain only finite values")

    if boxes.numel() > 0 and (
        torch.any(boxes < 0.0) or torch.any(boxes > 1.0)
    ):
        raise PairedTransformError(
            "normalized boxes must lie within [0, 1]"
        )


def apply_synchronized_geometry(
    rgb: Tensor,
    ir: Tensor,
    boxes: Tensor,
    *,
    horizontal_flip: bool = False,
    vertical_flip: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply identical geometric flips to RGB, IR, and normalized YOLO boxes."""

    if not isinstance(horizontal_flip, bool):
        raise PairedTransformError("horizontal_flip must be bool")

    if not isinstance(vertical_flip, bool):
        raise PairedTransformError("vertical_flip must be bool")

    _validate_inputs(rgb, ir, boxes)

    transformed_rgb = rgb.clone()
    transformed_ir = ir.clone()
    transformed_boxes = boxes.clone()

    if horizontal_flip:
        transformed_rgb = torch.flip(transformed_rgb, dims=(-1,))
        transformed_ir = torch.flip(transformed_ir, dims=(-1,))
        if transformed_boxes.numel() > 0:
            transformed_boxes[:, 0] = 1.0 - transformed_boxes[:, 0]

    if vertical_flip:
        transformed_rgb = torch.flip(transformed_rgb, dims=(-2,))
        transformed_ir = torch.flip(transformed_ir, dims=(-2,))
        if transformed_boxes.numel() > 0:
            transformed_boxes[:, 1] = 1.0 - transformed_boxes[:, 1]

    return transformed_rgb, transformed_ir, transformed_boxes

class DeterministicSynchronizedRandomFlip:
    """Choose paired flips from seed, epoch, split, and pair identity."""

    def __init__(
        self,
        *,
        seed: int,
        horizontal_probability: float = 0.5,
        vertical_probability: float = 0.0,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise PairedTransformError("seed must be an integer")

        self.seed = seed
        self.horizontal_probability = self._validate_probability(
            horizontal_probability
        )
        self.vertical_probability = self._validate_probability(
            vertical_probability
        )
        self.epoch = 0

    @staticmethod
    def _validate_probability(value: float) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise PairedTransformError(
                "flip probability must be finite and lie within [0, 1]"
            )
        return float(value)

    def set_epoch(self, epoch: int) -> None:
        if (
            isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or epoch < 0
        ):
            raise PairedTransformError(
                "epoch must be a non-negative integer"
            )
        self.epoch = epoch

    def _decisions(
        self,
        *,
        pair_id: str,
        split: str,
    ) -> tuple[bool, bool]:
        if not pair_id:
            raise PairedTransformError("pair_id must not be empty")

        if split not in {"train", "val"}:
            raise PairedTransformError(f"unsupported split: {split!r}")

        identity = (
            f"{self.seed}:{self.epoch}:{split}:{pair_id}"
        ).encode()
        digest = hashlib.sha256(identity).digest()

        horizontal_value = int.from_bytes(digest[:8], "big") / (1 << 64)
        vertical_value = int.from_bytes(digest[8:16], "big") / (1 << 64)

        return (
            horizontal_value < self.horizontal_probability,
            vertical_value < self.vertical_probability,
        )

    def __call__(
        self,
        rgb: Tensor,
        ir: Tensor,
        boxes: Tensor,
        *,
        pair_id: str,
        split: str,
    ) -> tuple[Tensor, Tensor, Tensor]:
        horizontal_flip, vertical_flip = self._decisions(
            pair_id=pair_id,
            split=split,
        )

        return apply_synchronized_geometry(
            rgb,
            ir,
            boxes,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
        )