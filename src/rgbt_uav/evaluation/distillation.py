from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from torchvision.ops import batched_nms


class DistillationInferenceError(ValueError):
    """Raised when student predictions cannot be decoded."""


def _validate_image_size(image_size: tuple[int, int]) -> tuple[int, int]:
    if (
        not isinstance(image_size, tuple)
        or len(image_size) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in image_size
        )
    ):
        raise DistillationInferenceError(
            "image_size must be a tuple of two positive integers"
        )

    return image_size


def decode_student_predictions(
    predictions: Sequence[Tensor],
    *,
    image_size: tuple[int, int],
    num_classes: int,
    reg_max: int,
    score_threshold: float = 0.25,
    iou_threshold: float = 0.7,
    max_det: int = 300,
) -> list[dict[str, Tensor]]:
    """Decode raw student pyramid outputs into per-image detections."""

    height, width = _validate_image_size(image_size)

    if not predictions:
        raise DistillationInferenceError(
            "predictions must contain at least one level"
        )

    if (
        isinstance(num_classes, bool)
        or not isinstance(num_classes, int)
        or num_classes <= 0
    ):
        raise DistillationInferenceError(
            "num_classes must be a positive integer"
        )

    if (
        isinstance(reg_max, bool)
        or not isinstance(reg_max, int)
        or reg_max <= 0
    ):
        raise DistillationInferenceError(
            "reg_max must be a positive integer"
        )

    if (
        isinstance(score_threshold, bool)
        or not isinstance(score_threshold, (int, float))
        or not 0.0 <= float(score_threshold) <= 1.0
    ):
        raise DistillationInferenceError(
            "score_threshold must be between 0 and 1"
        )

    if (
        isinstance(iou_threshold, bool)
        or not isinstance(iou_threshold, (int, float))
        or not 0.0 <= float(iou_threshold) <= 1.0
    ):
        raise DistillationInferenceError(
            "iou_threshold must be between 0 and 1"
        )

    if (
        isinstance(max_det, bool)
        or not isinstance(max_det, int)
        or max_det <= 0
    ):
        raise DistillationInferenceError(
            "max_det must be a positive integer"
        )

    expected_channels = 4 * reg_max + num_classes
    batch_size: int | None = None
    decoded: list[list[dict[str, Tensor]]] = []

    for level_index, level in enumerate(predictions):
        if not isinstance(level, Tensor) or level.ndim != 4:
            raise DistillationInferenceError(
                f"predictions[{level_index}] must have shape [B, C, H, W]"
            )

        if level.shape[1] != expected_channels:
            raise DistillationInferenceError(
                f"predictions[{level_index}] must have {expected_channels} channels"
            )

        if batch_size is None:
            batch_size = int(level.shape[0])
            if batch_size <= 0:
                raise DistillationInferenceError(
                    "prediction batch size must be positive"
                )
            decoded = [[] for _ in range(batch_size)]
        elif level.shape[0] != batch_size:
            raise DistillationInferenceError(
                "prediction levels must have equal batch sizes"
            )

        grid_height = int(level.shape[2])
        grid_width = int(level.shape[3])
        if grid_height <= 0 or grid_width <= 0:
            raise DistillationInferenceError(
                "prediction levels must have positive spatial dimensions"
            )

        class_logits = level[:, :num_classes]
        box_logits = level[:, num_classes:]

        # The paired smoke loss supervises the first four regression channels.
        # Keep decoding aligned with that training contract.
        box_logits = box_logits[:, :4].reshape(
            batch_size,
            4,
            grid_height,
            grid_width,
        )

        boxes_xywh = box_logits.sigmoid()
        class_scores = class_logits.sigmoid()

        local_maximum = class_scores == functional.max_pool2d(
            class_scores,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        suppressed_scores = class_scores.masked_fill(
            ~local_maximum,
            float("-inf"),
        )

        scores, labels = suppressed_scores.max(dim=1)

        keep = torch.isfinite(scores) & (scores >= float(score_threshold))

        yy, xx = torch.meshgrid(
            torch.arange(
                grid_height,
                device=level.device,
                dtype=level.dtype,
            ),
            torch.arange(
                grid_width,
                device=level.device,
                dtype=level.dtype,
            ),
            indexing="ij",
        )

        center_x = (xx + boxes_xywh[:, 0]) / grid_width * width

        center_y = (yy + boxes_xywh[:, 1]) / grid_height * height

        box_width = boxes_xywh[:, 2] * width
        box_height = boxes_xywh[:, 3] * height

        x1 = (center_x - box_width / 2).clamp(0, width)
        y1 = (center_y - box_height / 2).clamp(0, height)
        x2 = (center_x + box_width / 2).clamp(0, width)
        y2 = (center_y + box_height / 2).clamp(0, height)

        for image_index in range(batch_size):
            level_keep = keep[image_index] & (x2[image_index] > x1[image_index]) & (
                y2[image_index] > y1[image_index]
            )

            level_boxes = torch.stack(
                (
                    x1[image_index][level_keep],
                    y1[image_index][level_keep],
                    x2[image_index][level_keep],
                    y2[image_index][level_keep],
                ),
                dim=1,
            )
            level_scores = scores[image_index][level_keep]
            level_labels = labels[image_index][level_keep].to(dtype=torch.int64)

            if level_boxes.numel() > 0:
                keep_indices = batched_nms(
                    level_boxes,
                    level_scores,
                    level_labels,
                    float(iou_threshold),
                )
                keep_indices = keep_indices[:max_det]
                level_boxes = level_boxes[keep_indices]
                level_scores = level_scores[keep_indices]
                level_labels = level_labels[keep_indices]

            decoded[image_index].append(
                {
                    "boxes": level_boxes,
                    "scores": level_scores,
                    "labels": level_labels,
                }
            )

    assert batch_size is not None

    results: list[dict[str, Tensor]] = []

    for image_levels in decoded:
        if not image_levels:
            results.append(
                {
                    "boxes": torch.empty((0, 4)),
                    "scores": torch.empty((0,)),
                    "labels": torch.empty((0,), dtype=torch.int64),
                }
            )
            continue

        merged_boxes = torch.cat(
            [value["boxes"] for value in image_levels],
            dim=0,
        )
        merged_scores = torch.cat(
            [value["scores"] for value in image_levels],
            dim=0,
        )
        merged_labels = torch.cat(
            [value["labels"] for value in image_levels],
            dim=0,
        )

        if merged_boxes.numel() > 0:
            keep_indices = batched_nms(
                merged_boxes,
                merged_scores,
                merged_labels,
                float(iou_threshold),
            )
            keep_indices = keep_indices[:max_det]
            merged_boxes = merged_boxes[keep_indices]
            merged_scores = merged_scores[keep_indices]
            merged_labels = merged_labels[keep_indices]

        results.append(
            {
                "boxes": merged_boxes,
                "scores": merged_scores,
                "labels": merged_labels,
            }
        )

    return results

def run_distillation_student_inference(
    model: nn.Module,
    loader: Iterable[Mapping[str, object]],
    *,
    device: torch.device | str,
    num_classes: int,
    reg_max: int,
    score_threshold: float = 0.25,
) -> list[dict[str, object]]:
    """Run raw student inference and preserve paired sample IDs."""

    if not isinstance(model, nn.Module):
        raise DistillationInferenceError(
            "model must be a torch.nn.Module"
        )

    target_device = torch.device(device)
    was_training = model.training
    model.to(target_device)
    model.eval()

    results: list[dict[str, object]] = []

    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                if not isinstance(batch, Mapping):
                    raise DistillationInferenceError(
                        f"batch[{batch_index}] must be a mapping"
                    )

                rgb = batch.get("rgb")
                ir = batch.get("ir")
                pair_ids = batch.get("pair_ids")

                if not isinstance(rgb, Tensor) or rgb.ndim != 4:
                    raise DistillationInferenceError(
                        "batch rgb must have shape [B, 3, H, W]"
                    )

                if not isinstance(ir, Tensor) or ir.ndim != 4:
                    raise DistillationInferenceError(
                        "batch ir must have shape [B, 1, H, W]"
                    )

                if not isinstance(pair_ids, tuple):
                    raise DistillationInferenceError(
                        "batch pair_ids must be a tuple"
                    )

                if len(pair_ids) != rgb.shape[0]:
                    raise DistillationInferenceError(
                        "pair_ids length must match batch size"
                    )

                output = model(
                    rgb.to(target_device),
                    ir.to(target_device),
                )

                if not isinstance(output, Mapping):
                    raise DistillationInferenceError(
                        "student output must be a mapping"
                    )

                predictions = output.get("predictions")

                if not isinstance(predictions, (list, tuple)):
                    raise DistillationInferenceError("student predictions must be a sequence")

                if not predictions:
                    raise DistillationInferenceError("student predictions must not be empty")

                decoded = decode_student_predictions(
                    predictions,
                    image_size=(
                        int(rgb.shape[2]),
                        int(rgb.shape[3]),
                    ),
                    num_classes=num_classes,
                    reg_max=reg_max,
                    score_threshold=score_threshold,
                    iou_threshold=0.7,
                    max_det=300,
                )

                for pair_id, prediction in zip(
                    pair_ids,
                    decoded,
                    strict=True,
                ):
                    results.append(
                        {
                            "pair_id": pair_id,
                            **prediction,
                        }
                    )
    finally:
        model.train(was_training)

    return results