"""Real paired-teacher workload adapters for formal calibration."""

from __future__ import annotations

import gc
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from ultralytics.utils.loss import v8DetectionLoss

from rgbt_uav.data.paired import PairedYoloDataset, collate_paired_batch
from rgbt_uav.models.official_gated_teacher import (
    build_official_gated_teacher,
)
from rgbt_uav.training.early_fusion_config import EarlyFusionConfig
from rgbt_uav.training.early_fusion_pretrained import load_adapted_state_dict
from rgbt_uav.training.early_fusion_runner import _load_pretrained_state_dict
from rgbt_uav.training.early_fusion_runtime import (
    build_early_fusion_model,
    seed_early_fusion,
)
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
from rgbt_uav.training.official_gated_teacher_config import (
    OfficialGatedTeacherConfig,
)
from rgbt_uav.training.official_gated_teacher_loss import (
    OfficialGatedTeacherDetectionLoss,
)
from rgbt_uav.training.official_gated_teacher_pretrained import (
    load_official_gated_teacher_pretrained,
)
from rgbt_uav.training.official_gated_teacher_step import (
    run_official_gated_teacher_step,
)
from rgbt_uav.training.ultralytics_adapter import (
    run_ultralytics_early_fusion_step,
)


def _device_matches(configured: str, resolved: str) -> bool:
    value = configured.strip().lower()
    if value.isdigit():
        value = f"cuda:{value}"
    return value == resolved


def _validate_identity(
    workload: FormalCalibrationWorkload,
    *,
    expected_experiment: str,
    config_seed: int,
    config_device: str,
    resolved_device: str,
) -> None:
    if workload.experiment != expected_experiment:
        raise FormalCalibrationWorkloadError(
            f"expected {expected_experiment} workload"
        )
    if workload.seed != config_seed:
        raise FormalCalibrationWorkloadError(
            "workload seed does not match configuration"
        )
    if not _device_matches(config_device, resolved_device):
        raise FormalCalibrationWorkloadError(
            "calibration device does not match configuration"
        )


def _loader(
    dataset: Any,
    *,
    batch: int,
    workers: int,
    loader_factory: Callable[..., Any],
) -> Any:
    if len(dataset) <= 0:
        raise FormalCalibrationWorkloadError(
            "calibration training dataset must not be empty"
        )
    return loader_factory(
        dataset,
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_paired_batch,
    )


def _effective_batch(workload: FormalCalibrationWorkload, configured: int) -> int:
    batch = configured if workload.batch is None else workload.batch
    if isinstance(batch, bool) or not isinstance(batch, int) or batch <= 0:
        raise FormalCalibrationWorkloadError(
            "calibration batch override must be a positive integer"
        )
    return batch


def _resource_closer(resources: dict[str, Any], device: str) -> Callable[[], None]:
    def close() -> None:
        model = resources.get("model")
        optimizer = resources.get("optimizer")
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        if model is not None:
            model.to("cpu")
        resources.clear()
        gc.collect()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return close


def build_early_fusion_calibration_runtime(
    workload: FormalCalibrationWorkload,
    device: str,
    *,
    dataset_factory: Callable[..., Any] = PairedYoloDataset,
    loader_factory: Callable[..., Any] = DataLoader,
    model_factory: Callable[..., Any] = build_early_fusion_model,
    pretrained_state_loader: Callable[[Path], dict[str, torch.Tensor]] = (
        _load_pretrained_state_dict
    ),
    pretrained_loader: Callable[..., Any] = load_adapted_state_dict,
    criterion_factory: Callable[..., Any] = v8DetectionLoss,
    optimizer_factory: Callable[..., Any] = Adam,
    step_fn: Callable[..., Any] = run_ultralytics_early_fusion_step,
) -> FormalCalibrationRuntime:
    """Build the formal early-fusion teacher one-batch runtime."""

    config = EarlyFusionConfig.from_yaml(workload.config)
    _validate_identity(
        workload,
        expected_experiment="early_fusion_teacher",
        config_seed=config.seed,
        config_device=config.device,
        resolved_device=device,
    )
    if config.resume:
        raise FormalCalibrationWorkloadError(
            "calibration requires a fresh-run teacher configuration"
        )

    seed_early_fusion(config.seed)
    target_device = torch.device(device)
    dataset = dataset_factory(config.data_root, split="train")
    loader = _loader(
        dataset,
        batch=_effective_batch(workload, config.batch),
        workers=config.workers,
        loader_factory=loader_factory,
    )
    model = model_factory(
        model_name=config.model,
        num_classes=config.num_classes,
        device=target_device,
    )
    if config.pretrained is not None:
        state = pretrained_state_loader(config.pretrained)
        pretrained_loader(model, state, strict=False)
    criterion = criterion_factory(model)
    optimizer = optimizer_factory(model.parameters(), lr=config.lr)
    resources = {
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
    }

    def train_batch(batch: Any) -> object:
        return step_fn(
            resources["model"],
            resources["criterion"],
            resources["optimizer"],
            batch,
            device=target_device,
        )

    return build_loader_calibration_runtime(
        loader,
        train_batch=train_batch,
        close=_resource_closer(resources, device),
    )


def build_gated_teacher_calibration_runtime(
    workload: FormalCalibrationWorkload,
    device: str,
    *,
    dataset_factory: Callable[..., Any] = PairedYoloDataset,
    loader_factory: Callable[..., Any] = DataLoader,
    model_factory: Callable[..., Any] = build_official_gated_teacher,
    pretrained_loader: Callable[..., Any] = (
        load_official_gated_teacher_pretrained
    ),
    criterion_factory: Callable[..., Any] = OfficialGatedTeacherDetectionLoss,
    optimizer_factory: Callable[..., Any] = AdamW,
    step_fn: Callable[..., Any] = run_official_gated_teacher_step,
) -> FormalCalibrationRuntime:
    """Build the formal gated-teacher one-batch runtime."""

    config = OfficialGatedTeacherConfig.from_yaml(workload.config)
    _validate_identity(
        workload,
        expected_experiment="gated_teacher",
        config_seed=config.seed,
        config_device=config.device,
        resolved_device=device,
    )
    if config.resume:
        raise FormalCalibrationWorkloadError(
            "calibration requires a fresh-run teacher configuration"
        )

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    target_device = torch.device(device)
    dataset = dataset_factory(config.data_root, split="train")
    loader = _loader(
        dataset,
        batch=_effective_batch(workload, config.batch),
        workers=config.workers,
        loader_factory=loader_factory,
    )
    model = model_factory(
        model_name=config.model,
        num_classes=config.num_classes,
        device=target_device,
    )
    pretrained_loader(model, config.pretrained)
    criterion = criterion_factory(model)
    optimizer = optimizer_factory(
        model.parameters(),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )
    resources = {
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
    }

    def train_batch(batch: Any) -> object:
        return step_fn(
            resources["model"],
            resources["criterion"],
            resources["optimizer"],
            batch,
            device=target_device,
        )

    return build_loader_calibration_runtime(
        loader,
        train_batch=train_batch,
        close=_resource_closer(resources, device),
    )


def paired_teacher_calibration_adapters() -> dict[str, WorkloadAdapter]:
    """Return the implemented paired-teacher workload adapters."""

    return {
        "early_fusion_teacher": build_early_fusion_calibration_runtime,
        "gated_teacher": build_gated_teacher_calibration_runtime,
    }
