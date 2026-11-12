from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from rgbt_uav.data.paired import PairedYoloDataset, collate_paired_batch
from rgbt_uav.losses.detection import PairedDetectionLoss
from rgbt_uav.losses.distillation import DistillationLoss, FeatureDistiller
from rgbt_uav.models.detectors import StudentRGBTDetector, TeacherRGBTDetector
from rgbt_uav.training.distillation import train_distillation_epoch
from rgbt_uav.training.distillation_checkpoint import (
    load_distillation_checkpoint,
    save_distillation_checkpoint,
)
from rgbt_uav.training.distillation_config import DistillationConfig
from rgbt_uav.training.distillation_pretrained import (
    load_teacher_checkpoint,
)
from rgbt_uav.training.distillation_state import DistillationState
from rgbt_uav.training.early_fusion_config import EarlyFusionConfig
from rgbt_uav.training.official_distillation_teacher import (
    build_official_early_fusion_teacher,
    load_official_early_fusion_teacher_checkpoint,
)


class DistillationRunnerError(RuntimeError):
    """Raised when a distillation preflight cannot be started or resumed."""


@dataclass(frozen=True)
class DistillationRunResult:
    """Published artifacts and metrics from a distillation run."""

    best_weights: Path
    last_weights: Path
    state_path: Path
    completion_path: Path
    metrics: dict[str, float]
    completed_epochs: tuple[int, ...]


def build_teacher_model(
    *,
    num_classes: int,
    base_channels: int = 32,
    reg_max: int = 16,
    device: torch.device,
) -> torch.nn.Module:
    return TeacherRGBTDetector(
        num_classes=num_classes,
        base_channels=base_channels,
        reg_max=reg_max,
    ).to(device)


def build_student_model(
    *,
    num_classes: int,
    base_channels: int = 16,
    reg_max: int = 16,
    device: torch.device,
) -> torch.nn.Module:
    return StudentRGBTDetector(
        num_classes=num_classes,
        base_channels=base_channels,
        reg_max=reg_max,
    ).to(device)

def evaluate_distillation_loss(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    detection_loss: PairedDetectionLoss,
    loader: Any,
    *,
    device: torch.device,
) -> float:
    """Evaluate the student detection loss without updating either model."""

    del teacher

    student.to(device)
    was_training = student.training
    student.eval()

    total = 0.0
    batches = 0

    try:
        with torch.no_grad():
            for batch in loader:
                rgb = batch["rgb"].to(device)
                ir = batch["ir"].to(device)

                boxes = [
                    value.to(device)
                    for value in batch["boxes"]
                ]
                labels = [
                    value.to(device)
                    for value in batch["labels"]
                ]

                output = student(rgb, ir)

                if not isinstance(output, dict):
                    raise DistillationRunnerError(
                        "student validation output must be a mapping"
                    )

                predictions = output.get("predictions")

                if not isinstance(predictions, (list, tuple)):
                    raise DistillationRunnerError(
                        "student validation predictions must be a sequence"
                    )

                metrics = detection_loss(
                    predictions,
                    boxes,
                    labels,
                )

                value = metrics.get("total")

                if not isinstance(value, torch.Tensor):
                    raise DistillationRunnerError(
                        "student validation loss must be a tensor"
                    )

                if not torch.isfinite(value):
                    raise DistillationRunnerError(
                        "student validation loss must be finite"
                    )

                total += float(value.detach().cpu())
                batches += 1
    finally:
        if was_training:
            student.train()

    if batches == 0:
        raise DistillationRunnerError(
            "validation loader produced no batches"
        )

    return total / batches


def run_distillation_preflight(
    config: DistillationConfig | EarlyFusionConfig,
    *,
    dataset_factory: Callable[..., Any] = PairedYoloDataset,
) -> DistillationRunResult:
    """Run one or more teacher-student distillation epochs."""

    if not isinstance(config, (DistillationConfig, EarlyFusionConfig)):
        raise DistillationRunnerError("config must be a DistillationConfig or EarlyFusionConfig")

    teacher_kind = getattr(
        config,
        "teacher_kind",
        "custom",
    )
    teacher_model = getattr(
        config,
        "teacher_model",
        "yolov8n.yaml",
    )

    use_cuda = config.device not in {"cpu", "-1"} and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    output = Path(config.output_root)
    output.mkdir(parents=True, exist_ok=True)
    last_weights = output / "last.pt"
    best_weights = output / "best.pt"
    state_path = output / "run_state.json"
    completion_path = output / "completion.json"

    train_dataset = dataset_factory(Path(config.data_root), split="train")
    val_dataset = dataset_factory(Path(config.data_root), split="val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch,
        shuffle=True,
        num_workers=config.workers,
        collate_fn=collate_paired_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch,
        shuffle=False,
        num_workers=config.workers,
        collate_fn=collate_paired_batch,
    )

    teacher_base_channels = getattr(
        config,
        "teacher_base_channels",
        32,
    )
    student_base_channels = getattr(
        config,
        "student_base_channels",
        16,
    )
    reg_max = getattr(
        config,
        "reg_max",
        16,
    )

    student_channels = [
        student_base_channels * 4,
        student_base_channels * 8,
        student_base_channels * 16,
    ]

    if teacher_kind == "official_early_fusion":
        teacher = build_official_early_fusion_teacher(
            model_name=teacher_model,
            num_classes=config.num_classes,
            device=device,
        )
        teacher_channels = [64, 128, 256]
    else:
        teacher = build_teacher_model(
            num_classes=config.num_classes,
            base_channels=teacher_base_channels,
            reg_max=reg_max,
            device=device,
        )
        teacher_channels = [
            teacher_base_channels * 4,
            teacher_base_channels * 8,
            teacher_base_channels * 16,
        ]

    student = build_student_model(
        num_classes=config.num_classes,
        base_channels=student_base_channels,
        reg_max=reg_max,
        device=device,
    )

    feature_distiller = FeatureDistiller(
        student_channels=student_channels,
        teacher_channels=teacher_channels,
    ).to(device)

    teacher_checkpoint = getattr(
        config,
        "teacher_checkpoint",
        getattr(config, "pretrained", None),
    )

    if teacher_checkpoint is not None and not config.resume:
        if teacher_kind == "official_early_fusion":
            load_official_early_fusion_teacher_checkpoint(
                teacher_checkpoint,
                teacher=teacher,
                map_location=device,
            )
        else:
            load_teacher_checkpoint(
                teacher_checkpoint,
                model=teacher,
                map_location=device,
            )
            
    detection_loss = PairedDetectionLoss(
        num_classes=config.num_classes,
        reg_max=reg_max,
    )
    distillation_loss = DistillationLoss(
        feature_weight=getattr(config, "feature_weight", 1.0),
        logit_weight=getattr(config, "logit_weight", 1.0),
        temperature=getattr(config, "temperature", 2.0),
    )
    optimizer = torch.optim.Adam(
        list(student.parameters()) + list(feature_distiller.parameters()),
        lr=config.lr,
    )

    start_epoch = 1
    best_val_loss: float | None = None

    if config.resume:
        if not state_path.is_file():
            raise DistillationRunnerError(
                f"resume state does not exist: {state_path}"
            )

        state = DistillationState.from_json(state_path)

        if state.experiment_id != config.experiment_id:
            raise DistillationRunnerError(
                "resume state experiment_id does not match configuration"
            )

        if not state.last_checkpoint.is_file():
            raise DistillationRunnerError(
                f"resume checkpoint does not exist: {state.last_checkpoint}"
            )

        restored = load_distillation_checkpoint(
            state.last_checkpoint,
            teacher=teacher,
            student=student,
            feature_distiller=feature_distiller,
            optimizer=optimizer,
            map_location=device,
        )

        restored_epoch = restored.get("epoch")
        if not isinstance(restored_epoch, int):
            raise DistillationRunnerError(
                "checkpoint epoch must be an integer"
            )

        if restored_epoch != state.completed_epoch:
            raise DistillationRunnerError(
                "checkpoint epoch does not match run state"
            )

        start_epoch = state.completed_epoch + 1
        best_val_loss = state.best_val_loss

    all_epochs: list[int] = []
    for epoch in range(start_epoch, config.epochs + 1):
        train_metrics = train_distillation_epoch(
            teacher,
            student,
            detection_loss,
            distillation_loss,
            feature_distiller,
            optimizer,
            train_loader,
            device=device,
        )
        val_loss = float(evaluate_distillation_loss(
            teacher, student, detection_loss, val_loader, device=device
        ))
        metrics = {**train_metrics, "val_loss": val_loss}
        save_distillation_checkpoint(
            last_weights,
            teacher=teacher,
            student=student,
            feature_distiller=feature_distiller,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
        )
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            save_distillation_checkpoint(
                best_weights,
                teacher=teacher,
                student=student,
                feature_distiller=feature_distiller,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
            )
        all_epochs.append(epoch)
        DistillationState(
            experiment_id=config.experiment_id,
            status="running",
            completed_epoch=epoch,
            best_val_loss=best_val_loss,
            last_checkpoint=last_weights,
            best_checkpoint=best_weights,
        ).write_json(state_path)

    if not all_epochs:
        raise DistillationRunnerError("no epoch was executed")

    DistillationState(
        experiment_id=config.experiment_id,
        status="completed",
        completed_epoch=all_epochs[-1],
        best_val_loss=best_val_loss,
        last_checkpoint=last_weights,
        best_checkpoint=best_weights,
    ).write_json(state_path)
    teacher_checkpoint = getattr(
        config,
        "teacher_checkpoint",
        getattr(config, "pretrained", None),
    )

    payload = {
        "status": "completed",
        "experiment_id": config.experiment_id,
        "experiment_group": config.experiment_group,
        "config": {
            "epochs": config.epochs,
            "batch": config.batch,
            "imgsz": config.imgsz,
            "seed": config.seed,
            "device": config.device,
            "workers": config.workers,
            "lr": config.lr,
            "num_classes": config.num_classes,
            "resume": config.resume,
            "teacher_base_channels": getattr(
                config,
                "teacher_base_channels",
                32,
            ),
            "student_base_channels": getattr(
                config,
                "student_base_channels",
                16,
            ),
            "reg_max": getattr(config, "reg_max", 16),
            "feature_weight": getattr(config, "feature_weight", 1.0),
            "logit_weight": getattr(config, "logit_weight", 1.0),
            "temperature": getattr(config, "temperature", 2.0),
            "teacher_kind": teacher_kind,
            "teacher_model": teacher_model,
            "teacher_checkpoint": (
                str(teacher_checkpoint) if teacher_checkpoint is not None else None
            ),
        },
        "dataset": {
            "train_count": len(train_dataset),
            "val_count": len(val_dataset),
        },
        "completed_epochs": all_epochs,
        "all_completed_epochs": list(range(1, all_epochs[-1] + 1)),
        "best_weights": str(best_weights),
        "last_weights": str(last_weights),
        "state_path": str(state_path),
        "metrics": {"val_loss": float(best_val_loss)},
    }
    completion_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return DistillationRunResult(
        best_weights=best_weights,
        last_weights=last_weights,
        state_path=state_path,
        completion_path=completion_path,
        metrics={"val_loss": float(best_val_loss)},
        completed_epochs=tuple(all_epochs),
    )
