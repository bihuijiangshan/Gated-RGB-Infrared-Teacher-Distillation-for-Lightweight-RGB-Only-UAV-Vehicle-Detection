"""Native Ultralytics unimodal workload adapters for calibration."""

from __future__ import annotations

import gc
import os
import shutil
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

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
from rgbt_uav.training.formal_unimodal_config import FormalUnimodalConfig


def _normalized_device(value: str | int) -> str:
    normalized = str(value).strip().lower()
    if normalized.isdigit():
        return f"cuda:{normalized}"
    return normalized


def _build_native_trainer(
    config: FormalUnimodalConfig,
    device: str,
    scratch_root: Path,
) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise FormalCalibrationWorkloadError(
            "ultralytics is required for unimodal calibration"
        ) from exc

    source = YOLO(config.model)
    arguments = {
        "model": source.overrides["model"],
        "task": source.task,
        "mode": "train",
        "data": str(config.data).replace("\\", "/"),
        "epochs": 1,
        "imgsz": config.imgsz,
        "rect": config.rect,
        "batch": config.batch,
        "seed": config.seed,
        "workers": config.workers,
        "device": device,
        "amp": config.amp,
        "cache": config.cache,
        "optimizer": config.optimizer,
        "lr0": float(config.lr),
        "weight_decay": float(config.weight_decay),
        "project": str(scratch_root),
        "name": "trainer",
        "exist_ok": True,
        "save": False,
        "val": False,
        "plots": False,
        "verbose": False,
    }
    trainer_class = source._smart_load("trainer")
    trainer = trainer_class(
        overrides=arguments,
        _callbacks=source.callbacks,
    )
    trainer.model = trainer.get_model(
        weights=source.model if source.ckpt else None,
        cfg=source.model.yaml,
    )
    trainer._setup_train(world_size=1)
    return trainer


def build_unimodal_calibration_runtime(
    workload: FormalCalibrationWorkload,
    device: str,
    *,
    trainer_factory: Any = _build_native_trainer,
    scratch_parent: Path = Path(
        "runs/archives/formal_training_calibration_scratch"
    ),
) -> FormalCalibrationRuntime:
    """Build an AMP-aware RGB-only or IR-only native training runtime."""

    if workload.experiment not in {"rgb_only", "ir_only"}:
        raise FormalCalibrationWorkloadError(
            f"unsupported unimodal workload: {workload.experiment}"
        )
    config = FormalUnimodalConfig.from_yaml(workload.config)
    if config.modality != workload.experiment.removesuffix("_only"):
        raise FormalCalibrationWorkloadError(
            "workload modality does not match configuration"
        )
    if config.seed != workload.seed:
        raise FormalCalibrationWorkloadError(
            "workload seed does not match configuration"
        )
    if _normalized_device(config.device) != device:
        raise FormalCalibrationWorkloadError(
            "calibration device does not match configuration"
        )
    batch = config.batch if workload.batch is None else workload.batch
    if isinstance(batch, bool) or not isinstance(batch, int) or batch <= 0:
        raise FormalCalibrationWorkloadError(
            "calibration batch override must be a positive integer"
        )
    runtime_config = replace(config, batch=batch)

    scratch_parent = Path(scratch_parent)
    scratch_parent.mkdir(parents=True, exist_ok=True)
    scratch_root = scratch_parent / (
        f"{workload.experiment}-process-{os.getpid()}"
    )
    if scratch_root.exists() or scratch_root.is_symlink():
        raise FormalCalibrationWorkloadError(
            f"calibration scratch path already exists: {scratch_root}"
        )
    scratch_root.mkdir()
    try:
        trainer = trainer_factory(runtime_config, device, scratch_root)
        loader = trainer.train_loader
        model = trainer.model
        optimizer = trainer.optimizer
        scaler = trainer.scaler
        preprocess_batch = trainer.preprocess_batch
        optimizer_step = trainer.optimizer_step
        batch_size = int(trainer.batch_size)
        nominal_batch_size = int(trainer.args.nbs)
        use_amp = bool(trainer.amp)
    except Exception:
        shutil.rmtree(scratch_root, ignore_errors=True)
        raise

    if batch_size <= 0 or nominal_batch_size <= 0:
        shutil.rmtree(scratch_root, ignore_errors=True)
        raise FormalCalibrationWorkloadError(
            "trainer batch sizes must be positive"
        )
    accumulate = max(round(nominal_batch_size / batch_size), 1)
    state = {
        "trainer": trainer,
        "model": model,
        "optimizer": optimizer,
        "scaler": scaler,
    }
    batch_index = 0
    optimizer.zero_grad(set_to_none=True)
    model.train()

    def train_batch(batch: Any) -> object:
        nonlocal batch_index
        processed = preprocess_batch(batch)
        context = (
            torch.autocast(device_type=torch.device(device).type, enabled=True)
            if use_amp
            else nullcontext()
        )
        with context:
            loss, loss_items = model(processed)
        scaler.scale(loss).backward()
        batch_index += 1
        if batch_index % accumulate == 0:
            optimizer_step()
        return loss_items

    def close() -> None:
        try:
            optimizer.zero_grad(set_to_none=True)
            model.to("cpu")
            state.clear()
            gc.collect()
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            shutil.rmtree(scratch_root)

    return build_loader_calibration_runtime(
        loader,
        train_batch=train_batch,
        close=close,
    )


def unimodal_calibration_adapters() -> dict[str, WorkloadAdapter]:
    """Return the RGB-only and IR-only native workload adapters."""

    return {
        "rgb_only": build_unimodal_calibration_runtime,
        "ir_only": build_unimodal_calibration_runtime,
    }
