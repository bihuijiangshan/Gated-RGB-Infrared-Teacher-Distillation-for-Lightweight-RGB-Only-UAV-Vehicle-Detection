from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from rgbt_uav.data.paired import (
    PairedYoloDataset,
    collate_paired_batch,
)
from rgbt_uav.losses.detection import PairedDetectionLoss
from rgbt_uav.models.detectors import TeacherRGBTDetector
from rgbt_uav.training.distillation_config import DistillationConfig
from rgbt_uav.training.teacher_checkpoint import (
    save_teacher_training_checkpoint,
)
from rgbt_uav.training.teacher_epoch import train_teacher_epoch


class TeacherRunnerError(RuntimeError):
    """Raised when a teacher preflight cannot be completed."""


@dataclass(frozen=True)
class TeacherRunResult:
    """Published artifacts and metrics from teacher supervised training."""

    best_weights: Path
    last_weights: Path
    metrics: dict[str, float]
    completed_epochs: tuple[int, ...]


def build_teacher_model(
    *,
    num_classes: int,
    base_channels: int,
    reg_max: int,
    device: torch.device,
) -> torch.nn.Module:
    return TeacherRGBTDetector(
        num_classes=num_classes,
        base_channels=base_channels,
        reg_max=reg_max,
    ).to(device)


def evaluate_teacher_loss(
    model: torch.nn.Module,
    criterion: PairedDetectionLoss,
    loader: Any,
    *,
    device: torch.device,
) -> float:
    model.to(device)
    was_training = model.training
    model.eval()

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

                output = model(rgb, ir)
                predictions = output["predictions"]
                metrics = criterion(predictions, boxes, labels)
                value = metrics["total"]

                if not torch.isfinite(value):
                    raise TeacherRunnerError(
                        "teacher validation loss must be finite"
                    )

                total += float(value.detach().cpu())
                batches += 1
    finally:
        if was_training:
            model.train()

    if batches == 0:
        raise TeacherRunnerError(
            "teacher validation loader produced no batches"
        )

    return total / batches


def run_teacher_preflight(
    config: DistillationConfig,
    *,
    dataset_factory: Callable[..., Any] = PairedYoloDataset,
) -> TeacherRunResult:
    """Run a supervised teacher preflight."""

    if not isinstance(config, DistillationConfig):
        raise TeacherRunnerError(
            "config must be a DistillationConfig"
        )

    use_cuda = (
        config.device not in {"cpu", "-1"}
        and torch.cuda.is_available()
    )
    device = torch.device("cuda" if use_cuda else "cpu")

    output = Path(config.output_root)
    output.mkdir(parents=True, exist_ok=True)

    best_weights = output / "teacher-best.pt"
    last_weights = output / "teacher-last.pt"

    train_dataset = dataset_factory(
        Path(config.data_root),
        split="train",
    )
    val_dataset = dataset_factory(
        Path(config.data_root),
        split="val",
    )

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

    model = build_teacher_model(
        num_classes=config.num_classes,
        base_channels=config.teacher_base_channels,
        reg_max=config.reg_max,
        device=device,
    )

    criterion = PairedDetectionLoss(
        num_classes=config.num_classes,
        reg_max=config.reg_max,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
    )

    best_val_loss: float | None = None
    completed_epochs: list[int] = []
    final_metrics: dict[str, float] = {}

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_teacher_epoch(
            model,
            criterion,
            optimizer,
            train_loader,
            device=device,
        )
        val_loss = evaluate_teacher_loss(
            model,
            criterion,
            val_loader,
            device=device,
        )

        metrics = {
            **train_metrics,
            "val_loss": float(val_loss),
        }

        save_teacher_training_checkpoint(
            last_weights,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
        )

        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            save_teacher_training_checkpoint(
                best_weights,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
            )

        completed_epochs.append(epoch)
        final_metrics = metrics

    if not completed_epochs or best_val_loss is None:
        raise TeacherRunnerError(
            "no teacher epoch was executed"
        )

    return TeacherRunResult(
        best_weights=best_weights,
        last_weights=last_weights,
        metrics={
            key: float(value)
            for key, value in final_metrics.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        },
        completed_epochs=tuple(completed_epochs),
    )