"""Inference adapter for an official Ultralytics RGB student."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import Tensor, nn

from rgbt_uav.evaluation.paired import (
    ultralytics_decoded_to_paired_predictions,
)


class OfficialRGBStudentInferenceError(RuntimeError):
    """Raised when official RGB student inference fails."""


def run_official_rgb_student_inference(
    student: nn.Module,
    loader: Iterable[Mapping[str, object]],
    *,
    device: torch.device,
    score_threshold: float,
    iou_threshold: float,
    max_det: int,
) -> list[dict[str, Any]]:
    """Run decoded official YOLO inference using RGB images only."""

    if not isinstance(student, nn.Module):
        raise OfficialRGBStudentInferenceError(
            "student must be a module"
        )
    if not isinstance(device, torch.device):
        raise OfficialRGBStudentInferenceError(
            "device must be a torch.device"
        )

    model = getattr(student, "model", None)
    if not isinstance(model, nn.Module):
        raise OfficialRGBStudentInferenceError(
            "student must expose an official model"
        )

    student.to(device)
    student.eval()
    model.eval()

    results: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            rgb = batch.get("rgb")
            pair_ids = batch.get("pair_ids")
            if pair_ids is None:
                pair_ids = batch.get("pair_id")

            if not isinstance(rgb, Tensor):
                raise OfficialRGBStudentInferenceError(
                    f"batch {batch_index} rgb must be a tensor"
                )
            if isinstance(pair_ids, (str, bytes, Mapping)) or not isinstance(pair_ids, Iterable):
                raise OfficialRGBStudentInferenceError(
                    f"batch {batch_index} pair_ids must be iterable"
                )

            try:
                normalized_pair_ids = tuple(pair_ids)
            except TypeError as exc:
                raise OfficialRGBStudentInferenceError(
                    f"batch {batch_index} pair_ids cannot be normalized"
                ) from exc

            if not all(isinstance(pair_id, str) and pair_id for pair_id in normalized_pair_ids):
                raise OfficialRGBStudentInferenceError(
                    f"batch {batch_index} pair ids must be non-empty strings"
                )

            if len(normalized_pair_ids) != rgb.shape[0]:
                raise OfficialRGBStudentInferenceError(
                    f"batch {batch_index} pair ids do not match batch size"
                )

            output = model(
                rgb.to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
            )

            if (
                not isinstance(output, tuple)
                or len(output) != 2
            ):
                raise OfficialRGBStudentInferenceError(
                    "official model must return decoded and raw predictions"
                )

            decoded = output[0]
            if not isinstance(decoded, Tensor):
                raise OfficialRGBStudentInferenceError(
                    "decoded predictions must be a tensor"
                )

            try:
                batch_results = ultralytics_decoded_to_paired_predictions(
                    decoded,
                    pair_ids=normalized_pair_ids,
                    score_threshold=score_threshold,
                    iou_threshold=iou_threshold,
                    max_det=max_det,
                )
            except (RuntimeError, TypeError, ValueError) as exc:
                raise OfficialRGBStudentInferenceError(
                    f"failed to decode batch {batch_index}"
                ) from exc

            results.extend(batch_results)

    return results