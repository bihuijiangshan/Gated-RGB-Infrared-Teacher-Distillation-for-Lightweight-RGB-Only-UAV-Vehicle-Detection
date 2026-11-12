"""Lightweight RGB student workload adapters for formal calibration."""

from __future__ import annotations

import gc
import random
from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from rgbt_uav.data.paired import PairedYoloDataset, collate_paired_batch
from rgbt_uav.training.formal_calibration_plan import (
    FormalCalibrationWorkload,
)
from rgbt_uav.training.formal_calibration_runner import (
    FormalCalibrationRuntime,
)
from rgbt_uav.training.formal_calibration_workloads import (
    FormalCalibrationWorkloadError,
    WorkloadAdapter,
    build_loader_calibration_runtime,
)
from rgbt_uav.training.official_distillation_student import (
    build_official_rgb_student,
    load_official_rgb_student_pretrained,
    load_official_rgb_student_width_reduced_pretrained,
)
from rgbt_uav.training.official_rgb_distillation_config import (
    OfficialRGBDistillationConfig,
)
from rgbt_uav.training.official_rgb_distillation_runner import (
    _build_distillation_criterion,
    _configured_teacher_runtime,
    _load_configured_student_pretrained,
)
from rgbt_uav.training.official_student_loss import (
    OfficialStudentDetectionLoss,
)
from rgbt_uav.training.official_teacher_student_step import (
    run_official_teacher_student_step,
)

_EXPECTED_STUDENT_CONTRACTS = {
    "lightweight_detection_only": ("official_early_fusion", 0.0, 0.0),
    "lightweight_early_fusion_distillation": (
        "official_early_fusion",
        1.0,
        0.0,
    ),
    "lightweight_gated_distillation": ("official_gated", 1.0, 0.0),
}


def _normalized_device(value: str) -> str:
    normalized = value.strip().lower()
    if normalized.isdigit():
        return f"cuda:{normalized}"
    return normalized


def _effective_batch(workload: FormalCalibrationWorkload, configured: int) -> int:
    batch = configured if workload.batch is None else workload.batch
    if isinstance(batch, bool) or not isinstance(batch, int) or batch <= 0:
        raise FormalCalibrationWorkloadError(
            "calibration batch override must be a positive integer"
        )
    return batch


def _validate_contract(
    workload: FormalCalibrationWorkload,
    config: OfficialRGBDistillationConfig,
    device: str,
) -> None:
    expected = _EXPECTED_STUDENT_CONTRACTS.get(workload.experiment)
    if expected is None:
        raise FormalCalibrationWorkloadError(
            f"unsupported lightweight workload: {workload.experiment}"
        )
    teacher_kind, classification_weight, dfl_weight = expected
    if workload.seed != config.seed:
        raise FormalCalibrationWorkloadError(
            "workload seed does not match configuration"
        )
    if _normalized_device(config.device) != device:
        raise FormalCalibrationWorkloadError(
            "calibration device does not match configuration"
        )
    if config.resume:
        raise FormalCalibrationWorkloadError(
            "calibration requires a fresh-run student configuration"
        )
    if config.teacher_kind != teacher_kind:
        raise FormalCalibrationWorkloadError(
            "teacher_kind does not match lightweight workload"
        )
    calibration_teacher = workload.calibration_teacher
    if calibration_teacher is None:
        raise FormalCalibrationWorkloadError(
            "lightweight calibration requires a calibration teacher"
        )
    if calibration_teacher.kind != config.teacher_kind:
        raise FormalCalibrationWorkloadError(
            "calibration teacher kind does not match configuration"
        )
    if config.student_pretrained_kind != "width_reduced":
        raise FormalCalibrationWorkloadError(
            "lightweight calibration requires width_reduced pretrained loading"
        )
    if config.output_distillation_kind != "detection_aware":
        raise FormalCalibrationWorkloadError(
            "lightweight calibration requires detection_aware distillation"
        )
    if (
        float(config.feature_weight) != 0.0
        or config.effective_classification_distillation_weight
        != classification_weight
        or config.effective_dfl_distillation_weight != dfl_weight
    ):
        raise FormalCalibrationWorkloadError(
            "distillation weights do not match lightweight workload"
        )


def _close_resources(resources: dict[str, Any], device: str) -> None:
    optimizer = resources.get("optimizer")
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    for name in ("teacher", "student"):
        model = resources.get(name)
        if isinstance(model, nn.Module):
            model.to("cpu")
    resources.clear()
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_lightweight_student_calibration_runtime(
    workload: FormalCalibrationWorkload,
    device: str,
    *,
    dataset_factory: Callable[..., Any] = PairedYoloDataset,
    loader_factory: Callable[..., Any] = DataLoader,
    teacher_factory: Callable[..., Any] | None = None,
    teacher_checkpoint_loader: Callable[..., Any] | None = None,
    student_factory: Callable[..., Any] = build_official_rgb_student,
    exact_pretrained_loader: Callable[..., Any] = (
        load_official_rgb_student_pretrained
    ),
    width_reduced_pretrained_loader: Callable[..., Any] = (
        load_official_rgb_student_width_reduced_pretrained
    ),
    detection_criterion_factory: Callable[..., Any] = (
        OfficialStudentDetectionLoss
    ),
    distillation_criterion_factory: Callable[..., Any] | None = None,
    optimizer_factory: Callable[..., Any] = AdamW,
    step_fn: Callable[..., Any] = run_official_teacher_student_step,
) -> FormalCalibrationRuntime:
    """Build one of the three formal lightweight student runtimes."""

    config = OfficialRGBDistillationConfig.from_yaml(workload.config)
    _validate_contract(workload, config, device)

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    target_device = torch.device(device)

    dataset = dataset_factory(config.data_root, split="train")
    if len(dataset) <= 0:
        raise FormalCalibrationWorkloadError(
            "calibration training dataset must not be empty"
        )
    loader = loader_factory(
        dataset,
        batch_size=_effective_batch(workload, config.batch),
        shuffle=True,
        num_workers=config.workers,
        collate_fn=collate_paired_batch,
    )

    resolved_teacher_factory, resolved_checkpoint_loader = (
        _configured_teacher_runtime(
            config,
            teacher_factory=teacher_factory,
            teacher_checkpoint_loader=teacher_checkpoint_loader,
        )
    )
    teacher = resolved_teacher_factory(
        model_name=config.teacher_model,
        num_classes=config.num_classes,
        device=target_device,
    )
    resolved_checkpoint_loader(
        teacher,
        workload.calibration_teacher.checkpoint,
        map_location=target_device,
    )
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    student = student_factory(
        model_name=config.student_model,
        num_classes=config.num_classes,
        device=target_device,
    )
    _load_configured_student_pretrained(
        config,
        student,
        exact_loader=exact_pretrained_loader,
        width_reduced_loader=width_reduced_pretrained_loader,
    )
    detection_criterion = detection_criterion_factory(student)
    distillation_criterion = _build_distillation_criterion(
        config,
        distillation_criterion_factory,
    )
    optimizer = optimizer_factory(student.parameters(), lr=config.lr)
    resources = {
        "teacher": teacher,
        "student": student,
        "detection_criterion": detection_criterion,
        "distillation_criterion": distillation_criterion,
        "optimizer": optimizer,
    }

    def train_batch(batch: Any) -> object:
        return step_fn(
            resources["teacher"],
            resources["student"],
            resources["detection_criterion"],
            resources["distillation_criterion"],
            resources["optimizer"],
            batch,
            device=target_device,
            num_classes=config.num_classes,
        )

    return build_loader_calibration_runtime(
        loader,
        train_batch=train_batch,
        close=lambda: _close_resources(resources, device),
    )


def lightweight_student_calibration_adapters() -> dict[str, WorkloadAdapter]:
    """Return adapters for all three lightweight student workloads."""

    return {
        experiment: build_lightweight_student_calibration_runtime
        for experiment in _EXPECTED_STUDENT_CONTRACTS
    }
