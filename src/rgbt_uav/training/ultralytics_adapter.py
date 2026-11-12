from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


class UltralyticsAdapterError(ValueError):
    """Raised when a paired batch cannot be converted to Ultralytics format."""


def _require_tensor(
    batch: Mapping[str, Any],
    key: str,
) -> Tensor:
    value = batch.get(key)
    if not isinstance(value, Tensor):
        raise UltralyticsAdapterError(
            f"batch field {key!r} must be a tensor"
        )
    return value


def to_ultralytics_early_fusion_batch(
    batch: Mapping[str, Any],
) -> dict[str, Tensor | tuple[str, ...]]:
    """Convert a paired batch to the standard Ultralytics detection schema."""

    rgb = _require_tensor(batch, "rgb")
    ir = _require_tensor(batch, "ir")

    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise UltralyticsAdapterError(
            "rgb must have shape [batch, 3, height, width]"
        )

    if ir.ndim != 4 or ir.shape[1] != 1:
        raise UltralyticsAdapterError(
            "ir must have shape [batch, 1, height, width]"
        )

    if rgb.shape[0] != ir.shape[0] or rgb.shape[2:] != ir.shape[2:]:
        raise UltralyticsAdapterError(
            "rgb and ir batch/spatial shapes must match"
        )

    boxes = batch.get("boxes")
    labels = batch.get("labels")
    pair_ids = batch.get("pair_ids")

    if not isinstance(boxes, list) or not isinstance(labels, list):
        raise UltralyticsAdapterError(
            "boxes and labels must be lists"
        )

    if len(boxes) != rgb.shape[0] or len(labels) != rgb.shape[0]:
        raise UltralyticsAdapterError(
            "target list length must match image batch size"
        )

    if not isinstance(pair_ids, tuple) or len(pair_ids) != rgb.shape[0]:
        raise UltralyticsAdapterError(
            "pair_ids must match image batch size"
        )

    images = torch.cat((rgb, ir), dim=1)

    batch_indices: list[Tensor] = []
    all_boxes: list[Tensor] = []
    all_labels: list[Tensor] = []

    for image_index, (sample_boxes, sample_labels) in enumerate(
        zip(boxes, labels, strict=True)
    ):
        if not isinstance(sample_boxes, Tensor):
            raise UltralyticsAdapterError(
                f"boxes[{image_index}] must be a tensor"
            )

        if not isinstance(sample_labels, Tensor):
            raise UltralyticsAdapterError(
                f"labels[{image_index}] must be a tensor"
            )

        if sample_boxes.ndim != 2 or sample_boxes.shape[1] != 4:
            raise UltralyticsAdapterError(
                f"boxes[{image_index}] must have shape [num_boxes, 4]"
            )

        if sample_labels.ndim != 1:
            raise UltralyticsAdapterError(
                f"labels[{image_index}] must have shape [num_boxes]"
            )

        if sample_boxes.shape[0] != sample_labels.shape[0]:
            raise UltralyticsAdapterError(
                f"boxes and labels count differ at image {image_index}"
            )

        count = sample_boxes.shape[0]

        if count:
            batch_indices.append(
                torch.full(
                    (count,),
                    image_index,
                    dtype=torch.int64,
                    device=images.device,
                )
            )
            all_boxes.append(
                sample_boxes.to(
                    device=images.device,
                    dtype=torch.float32,
                )
            )
            all_labels.append(
                sample_labels.to(
                    device=images.device,
                    dtype=torch.float32,
                )
            )

    if batch_indices:
        merged_batch_indices = torch.cat(batch_indices, dim=0)
        merged_boxes = torch.cat(all_boxes, dim=0)
        merged_labels = torch.cat(all_labels, dim=0)
    else:
        merged_batch_indices = torch.empty(
            (0,),
            dtype=torch.int64,
            device=images.device,
        )
        merged_boxes = torch.empty(
            (0, 4),
            dtype=torch.float32,
            device=images.device,
        )
        merged_labels = torch.empty(
            (0,),
            dtype=torch.float32,
            device=images.device,
        )

    return {
        "img": images,
        "batch_idx": merged_batch_indices,
        "cls": merged_labels,
        "bboxes": merged_boxes,
        "pair_ids": pair_ids,
    }

def run_ultralytics_early_fusion_step(
    model: nn.Module,
    criterion: Any,
    optimizer: Optimizer,
    paired_batch: Mapping[str, Any],
    *,
    device: torch.device | str | None = None,
) -> dict[str, float]:
    """Run one official Ultralytics loss update on a paired batch."""

    if device is None:
        try:
            target_device = next(model.parameters()).device
        except StopIteration as exc:
            raise UltralyticsAdapterError(
                "model must contain parameters"
            ) from exc
    else:
        target_device = torch.device(device)

    model.to(target_device)
    model.train()

    converted = to_ultralytics_early_fusion_batch(paired_batch)

    images = converted["img"].to(target_device)
    batch_indices = converted["batch_idx"].to(target_device)
    classes = converted["cls"].to(target_device)
    boxes = converted["bboxes"].to(target_device)

    train_batch = {
        "img": images,
        "batch_idx": batch_indices,
        "cls": classes,
        "bboxes": boxes,
    }

    predictions = model(images)
    loss, loss_items = criterion(predictions, train_batch)

    if loss_items.numel() != 3:
        raise UltralyticsAdapterError(
            "official detection loss must return box, cls, and dfl items"
        )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return {
        "total": float(loss.detach().cpu()),
        "box": float(loss_items[0].detach().cpu()),
        "classification": float(loss_items[1].detach().cpu()),
        "dfl": float(loss_items[2].detach().cpu()),
    }