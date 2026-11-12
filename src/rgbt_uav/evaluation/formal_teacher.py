"""COCO evaluation for formal early-fusion and gated teachers."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader


class FormalTeacherEvaluationError(RuntimeError):
    """Raised when a formal teacher checkpoint cannot be evaluated."""


CLASS_NAMES = ("car", "truck", "bus", "van", "freight_car")
LABEL_TO_CATEGORY_ID = {index: index + 1 for index in range(len(CLASS_NAMES))}
CATEGORY_NAMES = {index + 1: name for index, name in enumerate(CLASS_NAMES)}


def _device(value: str) -> torch.device:
    normalized = value.strip().lower()
    if normalized == "cpu":
        return torch.device("cpu")
    device = torch.device(
        f"cuda:{normalized}" if normalized.isdigit() else normalized
    )
    if device.type != "cuda" or not torch.cuda.is_available():
        raise FormalTeacherEvaluationError("requested CUDA device is unavailable")
    index = 0 if device.index is None else device.index
    if index >= torch.cuda.device_count():
        raise FormalTeacherEvaluationError("requested CUDA device does not exist")
    return torch.device(f"cuda:{index}")


def _checkpoint(path: Path, device: torch.device) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FormalTeacherEvaluationError(
            f"checkpoint must be a regular non-symlink file: {path}"
        )
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise FormalTeacherEvaluationError(f"failed to load checkpoint: {path}") from exc
    if not isinstance(payload, Mapping):
        raise FormalTeacherEvaluationError("checkpoint root must be a mapping")
    epoch = payload.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
        raise FormalTeacherEvaluationError("checkpoint epoch must be positive")
    return payload


def _decoded(output: object) -> Tensor:
    value = output[0] if isinstance(output, (tuple, list)) and output else output
    if not isinstance(value, Tensor) or value.ndim != 3:
        raise FormalTeacherEvaluationError("decoded predictions must be a 3D tensor")
    return value


def _pair_ids(batch: Mapping[str, object]) -> tuple[str, ...]:
    value = batch.get("pair_ids", batch.get("pair_id"))
    if isinstance(value, str):
        return (value,)
    if value is None:
        raise FormalTeacherEvaluationError("batch is missing pair IDs")
    try:
        result = tuple(str(item) for item in value)  # type: ignore[union-attr]
    except TypeError as exc:
        raise FormalTeacherEvaluationError("batch pair IDs must be iterable") from exc
    return result


def _early_fusion_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    score_threshold: float,
    iou_threshold: float,
    max_det: int,
) -> list[dict[str, object]]:
    from rgbt_uav.evaluation.paired import (
        ultralytics_decoded_to_paired_predictions,
    )

    results: list[dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            rgb = batch.get("rgb")
            ir = batch.get("ir")
            if not isinstance(rgb, Tensor) or not isinstance(ir, Tensor):
                raise FormalTeacherEvaluationError("paired images must be tensors")
            pair_ids = _pair_ids(batch)
            if len(pair_ids) != rgb.shape[0] or ir.shape[0] != rgb.shape[0]:
                raise FormalTeacherEvaluationError("paired batch sizes do not match")
            fused = torch.cat(
                (
                    rgb.to(device=device, dtype=torch.float32, non_blocking=True),
                    ir.to(device=device, dtype=torch.float32, non_blocking=True),
                ),
                dim=1,
            )
            results.extend(
                ultralytics_decoded_to_paired_predictions(
                    _decoded(model(fused)),
                    pair_ids=pair_ids,
                    score_threshold=score_threshold,
                    iou_threshold=iou_threshold,
                    max_det=max_det,
                )
            )
    return results


def evaluate_formal_teacher_checkpoint(
    *,
    kind: str,
    checkpoint: Path,
    output: Path,
    model_name: str,
    data_root: Path,
    batch: int,
    workers: int,
    device: str,
    score_threshold: float,
    iou_threshold: float,
    max_det: int,
) -> dict[str, object]:
    """Evaluate one retained teacher epoch on the validation split."""

    if kind not in {"early_fusion", "gated"}:
        raise FormalTeacherEvaluationError("kind must be early_fusion or gated")
    if batch <= 0 or workers < 0 or max_det <= 0:
        raise FormalTeacherEvaluationError("evaluation limits are invalid")
    if not 0.0 <= score_threshold <= 1.0 or not 0.0 < iou_threshold <= 1.0:
        raise FormalTeacherEvaluationError("evaluation thresholds are invalid")

    from rgbt_uav.data.paired import PairedYoloDataset, collate_paired_batch
    from rgbt_uav.evaluation.paired import evaluate_paired_coco

    resolved_device = _device(device)
    checkpoint = Path(checkpoint)
    payload = _checkpoint(checkpoint, resolved_device)
    dataset = PairedYoloDataset(Path(data_root), split="val")
    loader = DataLoader(
        dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_paired_batch,
        pin_memory=resolved_device.type == "cuda",
    )

    if kind == "early_fusion":
        from rgbt_uav.training.early_fusion_runtime import build_early_fusion_model

        model = build_early_fusion_model(
            model_name=model_name,
            num_classes=len(CLASS_NAMES),
            device=resolved_device,
        )
        state = payload.get("model")
        if not isinstance(state, Mapping):
            raise FormalTeacherEvaluationError("checkpoint model state is missing")
        model.load_state_dict(state, strict=True)
        predictions = _early_fusion_predictions(
            model,
            loader,
            device=resolved_device,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            max_det=max_det,
        )
    else:
        from rgbt_uav.evaluation.official_gated_teacher import (
            run_official_gated_teacher_inference,
        )
        from rgbt_uav.models.official_gated_teacher import (
            build_official_gated_teacher,
        )

        model = build_official_gated_teacher(
            model_name=model_name,
            num_classes=len(CLASS_NAMES),
            device=resolved_device,
        )
        state = payload.get("teacher")
        if not isinstance(state, Mapping):
            raise FormalTeacherEvaluationError("checkpoint teacher state is missing")
        model.load_state_dict(state, strict=True)
        predictions = run_official_gated_teacher_inference(
            model,
            loader,
            device=resolved_device,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            max_det=max_det,
        )

    samples = (dataset[index] for index in range(len(dataset)))
    metrics = evaluate_paired_coco(
        samples,
        predictions,
        label_to_category_id=LABEL_TO_CATEGORY_ID,
        category_names=CATEGORY_NAMES,
    )
    report: dict[str, object] = {
        "format_version": 1,
        "status": "completed",
        "kind": kind,
        "device": str(resolved_device),
        "checkpoint": str(checkpoint),
        "model": model_name,
        "restored_epoch": int(payload["epoch"]),
        "split": "val",
        "test_evaluated": False,
        "images": len(predictions),
        "prediction_boxes": sum(len(item["boxes"]) for item in predictions),
        "mAP50-95": metrics.ap,
        "mAP50": metrics.ap50,
        "AP75": metrics.ap75,
        "AP_small": metrics.ap_small,
        "AR_small": metrics.ar_small,
    }
    if not all(
        isinstance(report[name], float) and math.isfinite(report[name])
        for name in ("mAP50-95", "mAP50", "AP75", "AP_small", "AR_small")
    ):
        raise FormalTeacherEvaluationError("COCO metrics must be finite")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
