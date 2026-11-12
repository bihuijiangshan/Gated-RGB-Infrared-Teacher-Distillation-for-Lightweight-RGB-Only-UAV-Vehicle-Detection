"""Resumable runner for official RGB-T to RGB distillation."""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from rgbt_uav.data.paired import (
    PairedYoloDataset,
    collate_paired_batch,
)
from rgbt_uav.losses.distillation import DistillationLoss
from rgbt_uav.losses.official_detection_distillation import OfficialTeacherStudentDistillationLoss
from rgbt_uav.models.official_gated_teacher import (
    build_official_gated_teacher,
)
from rgbt_uav.training.official_distillation_student import (
    build_official_rgb_student,
    load_official_rgb_student_pretrained,
    load_official_rgb_student_width_reduced_pretrained,
)
from rgbt_uav.training.official_distillation_teacher import (
    build_official_early_fusion_teacher,
)
from rgbt_uav.training.official_gated_teacher_checkpoint import (
    load_official_gated_teacher_weights,
)
from rgbt_uav.training.official_rgb_distillation_checkpoint import (
    load_official_rgb_distillation_checkpoint,
    save_official_rgb_distillation_checkpoint,
)
from rgbt_uav.training.official_rgb_distillation_config import (
    OfficialRGBDistillationConfig,
)
from rgbt_uav.training.official_student_loss import (
    OfficialStudentDetectionLoss,
)
from rgbt_uav.training.official_teacher_student_epoch import (
    train_official_teacher_student_epoch,
    validate_official_rgb_student,
)


class OfficialRGBDistillationRunnerError(RuntimeError):
    """Raised when the official RGB distillation run fails."""


@dataclass(frozen=True)
class OfficialRGBDistillationResult:
    best_weights: Path
    last_weights: Path
    state_path: Path
    completion_path: Path
    completed_epochs: tuple[int, ...]
    metrics: dict[str, float]


def _resolve_device(value: str) -> torch.device:
    normalized = value.strip().lower()

    if normalized == "cpu":
        return torch.device("cpu")

    if normalized.startswith("cuda"):
        device = torch.device(normalized)
    elif normalized.isdigit():
        device = torch.device(f"cuda:{normalized}")
    else:
        raise OfficialRGBDistillationRunnerError(
            f"unsupported device: {value}"
        )

    if not torch.cuda.is_available():
        raise OfficialRGBDistillationRunnerError(
            "CUDA was requested but is unavailable"
        )

    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_json(
    path: Path,
    payload: dict[str, object],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    try:
        temporary.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except (OSError, TypeError, ValueError) as exc:
        if temporary.is_file():
            temporary.unlink()
        raise OfficialRGBDistillationRunnerError(
            f"failed to write JSON evidence: {path}"
        ) from exc

    return path


def _load_teacher_checkpoint(
    teacher: nn.Module,
    checkpoint: Path,
    *,
    map_location: torch.device,
) -> dict[str, object]:
    if not checkpoint.is_file():
        raise OfficialRGBDistillationRunnerError(
            f"teacher checkpoint must be a regular file: {checkpoint}"
        )

    try:
        payload = torch.load(
            checkpoint,
            map_location=map_location,
            weights_only=False,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise OfficialRGBDistillationRunnerError(
            "failed to load teacher checkpoint"
        ) from exc

    if not isinstance(payload, dict):
        raise OfficialRGBDistillationRunnerError(
            "teacher checkpoint must be a dictionary"
        )
    if "model" not in payload:
        raise OfficialRGBDistillationRunnerError(
            "teacher checkpoint is missing model state"
        )

    wrapped_model = getattr(teacher, "model", None)
    if not isinstance(wrapped_model, nn.Module):
        raise OfficialRGBDistillationRunnerError(
            "teacher does not expose an official model"
        )

    try:
        wrapped_model.load_state_dict(
            payload["model"],
            strict=True,
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise OfficialRGBDistillationRunnerError(
            "teacher checkpoint is incompatible"
        ) from exc

    epoch = payload.get("epoch", -1)
    if not isinstance(epoch, int):
        raise OfficialRGBDistillationRunnerError(
            "teacher checkpoint epoch must be an integer"
        )

    return {"epoch": epoch}

def _load_gated_teacher_checkpoint(
    teacher: nn.Module,
    checkpoint: Path,
    *,
    map_location: torch.device,
) -> dict[str, object]:
    """Adapt the gated checkpoint loader to the runner contract."""

    try:
        return (
            load_official_gated_teacher_weights(
                checkpoint,
                teacher=teacher,
                map_location=map_location,
            )
        )
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise OfficialRGBDistillationRunnerError(
            "failed to load official gated "
            "teacher checkpoint"
        ) from exc


def _configured_teacher_runtime(
    config: OfficialRGBDistillationConfig,
    *,
    teacher_factory: (
        Callable[..., Any] | None
    ),
    teacher_checkpoint_loader: (
        Callable[..., Any] | None
    ),
) -> tuple[
    Callable[..., Any],
    Callable[..., Any],
]:
    """Resolve the configured teacher implementation and loader."""

    if (
        teacher_factory is None
        and teacher_checkpoint_loader is None
    ):
        if (
            config.teacher_kind
            == "official_early_fusion"
        ):
            return (
                build_official_early_fusion_teacher,
                _load_teacher_checkpoint,
            )

        if config.teacher_kind == "official_gated":
            return (
                build_official_gated_teacher,
                _load_gated_teacher_checkpoint,
            )

        raise OfficialRGBDistillationRunnerError(
            "unsupported teacher_kind: "
            f"{config.teacher_kind}"
        )

    if (
        teacher_factory is None
        or teacher_checkpoint_loader is None
    ):
        raise OfficialRGBDistillationRunnerError(
            "teacher factory and checkpoint "
            "loader must be overridden together"
        )

    return (
        teacher_factory,
        teacher_checkpoint_loader,
    )


def _optimizer_factory(
    parameters: Any,
    lr: float,
) -> AdamW:
    return AdamW(parameters, lr=lr)

def _load_configured_student_pretrained(
    config: OfficialRGBDistillationConfig,
    student: nn.Module,
    *,
    exact_loader: Callable[..., Any],
    width_reduced_loader: Callable[..., Any],
) -> dict[str, object]:
    if config.student_pretrained_kind == "exact":
        result = exact_loader(
            student,
            config.student_pretrained,
        )
    elif config.student_pretrained_kind == "width_reduced":
        result = width_reduced_loader(
            student,
            config.student_pretrained,
        )
    else:
        raise OfficialRGBDistillationRunnerError(
            "unsupported student_pretrained_kind: "
            f"{config.student_pretrained_kind}"
        )

    if not isinstance(result, dict):
        raise OfficialRGBDistillationRunnerError(
            "student pretrained loader must return a dictionary"
        )

    return result

def _build_distillation_criterion(
    config: OfficialRGBDistillationConfig,
    factory: Callable[..., Any] | None = None,
) -> Any:
    """Build the configured output-distillation implementation."""

    criterion_factory = factory

    if criterion_factory is None:
        if (
            config.output_distillation_kind
            == "legacy_joint"
        ):
            criterion_factory = DistillationLoss
        elif (
            config.output_distillation_kind
            == "detection_aware"
        ):
            criterion_factory = (
                OfficialTeacherStudentDistillationLoss
            )
        else:
            raise OfficialRGBDistillationRunnerError(
                "unsupported output distillation kind: "
                f"{config.output_distillation_kind}"
            )

    arguments: dict[str, object] = {
        "feature_weight": config.feature_weight,
        "logit_weight": config.logit_weight,
        "temperature": config.temperature,
    }

    if config.output_distillation_kind == "detection_aware":
        arguments.update(
            {
                "classification_distillation_weight": (
                    config.effective_classification_distillation_weight
                ),
                "dfl_distillation_weight": (config.effective_dfl_distillation_weight),
            }
        )

    return criterion_factory(**arguments)

def run_official_rgb_distillation(
    config: OfficialRGBDistillationConfig,
    *,
    dataset_factory: Callable[..., Any] = PairedYoloDataset,
    loader_factory: Callable[..., Any] = DataLoader,
    teacher_factory: (Callable[..., Any] | None) = None,
    student_factory: Callable[..., Any] = (build_official_rgb_student),
    teacher_checkpoint_loader: (Callable[..., Any] | None) = None,
    student_pretrained_loader: Callable[..., Any] = (load_official_rgb_student_pretrained),
    student_width_reduced_pretrained_loader: (Callable[..., Any]) = (
        load_official_rgb_student_width_reduced_pretrained
    ),
    detection_criterion_factory: Callable[..., Any] = (OfficialStudentDetectionLoss),
    distillation_criterion_factory: (Callable[..., Any] | None) = None,
    optimizer_factory: Callable[..., Any] = _optimizer_factory,
    train_epoch_fn: Callable[..., Any] = (train_official_teacher_student_epoch),
    validate_epoch_fn: Callable[..., Any] = (validate_official_rgb_student),
    checkpoint_saver: Callable[..., Any] = (save_official_rgb_distillation_checkpoint),
    checkpoint_loader: Callable[..., Any] = (load_official_rgb_distillation_checkpoint),
) -> OfficialRGBDistillationResult:
    """Run a fresh official RGB student distillation experiment."""

    if not isinstance(config, OfficialRGBDistillationConfig):
        raise OfficialRGBDistillationRunnerError(
            "config must be an OfficialRGBDistillationConfig"
        )
    output_root = config.output_root
    best_weights = output_root / "best.pt"
    last_weights = output_root / "last.pt"
    state_path = output_root / "run_state.json"
    completion_path = output_root / "completion.json"

    completed_before = 0
    best_val_loss = float("inf")
    epoch_weights: list[Path] = []
    previous_student_metadata = {
        "pretrained_loaded_keys": 0,
        "pretrained_exact_keys": 0,
        "pretrained_sliced_keys": 0,
        "pretrained_skipped_classification_keys": 0,
        "pretrained_missing_keys": 0,
    }

    if config.resume:
        for required in (
            state_path,
            completion_path,
            last_weights,
            best_weights,
        ):
            if not required.is_file():
                raise OfficialRGBDistillationRunnerError(
                    f"resume artifact must be a regular file: {required}"
                )

        try:
            state_payload = json.loads(state_path.read_text(encoding="utf-8"))
            completion_payload = json.loads(completion_path.read_text(encoding="utf-8"))
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
        ) as exc:
            raise OfficialRGBDistillationRunnerError("failed to read resume evidence") from exc

        if not isinstance(state_payload, dict):
            raise OfficialRGBDistillationRunnerError("run state must be a dictionary")
        if state_payload.get("experiment_id") != config.experiment_id:
            raise OfficialRGBDistillationRunnerError("resume experiment_id does not match")

        completed_before = state_payload.get("completed_epoch")
        if not isinstance(completed_before, int) or completed_before <= 0:
            raise OfficialRGBDistillationRunnerError("resume completed_epoch must be positive")
        if completed_before >= config.epochs:
            raise OfficialRGBDistillationRunnerError("resume target must exceed completed epoch")

        best_val_loss = float(state_payload.get("best_val_loss"))
        if not torch.isfinite(torch.tensor(best_val_loss)):
            raise OfficialRGBDistillationRunnerError("resume best_val_loss must be finite")

        if isinstance(completion_payload, dict):
            completion_config = completion_payload.get(
                "config"
            )

            if completion_config is None:
                # Early completion evidence did not contain a
                # configuration mapping. Those runs all used the
                # original joint-channel distillation loss.
                previous_kind = "legacy_joint"
            elif isinstance(completion_config, dict):
                previous_kind = completion_config.get(
                    "output_distillation_kind",
                    "legacy_joint",
                )
            else:
                raise OfficialRGBDistillationRunnerError(
                    "resume completion config must be a mapping"
                )
            if (
                    previous_kind
                    != config.output_distillation_kind
            ):
                raise OfficialRGBDistillationRunnerError(
                    "resume output_distillation_kind "
                    "does not match"
                )

            if completion_config is None:
                previous_pretrained_kind = "exact"
            else:
                previous_pretrained_kind = (
                    completion_config.get(
                        "student_pretrained_kind",
                        "exact",
                    )
                )

            if completion_config is None:
                previous_teacher_kind = (
                    "official_early_fusion"
                )
            else:
                previous_teacher_kind = (
                    completion_config.get(
                        "teacher_kind",
                        "official_early_fusion",
                    )
                )

            if (
                previous_teacher_kind
                != config.teacher_kind
            ):
                raise (
                    OfficialRGBDistillationRunnerError(
                        "resume teacher_kind "
                        "does not match"
                    )
                )

            if (
                previous_pretrained_kind
                != config.student_pretrained_kind
            ):
                raise OfficialRGBDistillationRunnerError(
                    "resume student_pretrained_kind "
                    "does not match"
                )

            student_metadata = completion_payload.get(
                "student"
            )
            previous_epoch_weights = (
                completion_payload.get(
                    "epoch_weights",
                    []
                )
            )
            if not isinstance(
                previous_epoch_weights,
                list,
            ):
                raise OfficialRGBDistillationRunnerError(
                    "resume epoch_weights must be a list"
                )

            for value in previous_epoch_weights:
                if not isinstance(value, str):
                    raise OfficialRGBDistillationRunnerError(
                        "resume epoch checkpoint path is invalid"
                    )

                epoch_path = Path(value)
                if not epoch_path.is_file():
                    raise OfficialRGBDistillationRunnerError(
                        "resume epoch checkpoint is missing: "
                        f"{epoch_path}"
                    )

                epoch_weights.append(epoch_path)
            if isinstance(student_metadata, dict):
                previous_student_metadata = {
                    "pretrained_loaded_keys": int(
                        student_metadata.get(
                            "pretrained_loaded_keys",
                            0,
                        )
                    ),
                    "pretrained_exact_keys": int(
                        student_metadata.get(
                            "pretrained_exact_keys",
                            0,
                        )
                    ),
                    "pretrained_sliced_keys": int(
                        student_metadata.get(
                            "pretrained_sliced_keys",
                            0,
                        )
                    ),
                    "pretrained_skipped_classification_keys": int(
                        student_metadata.get(
                            "pretrained_skipped_classification_keys",
                            0,
                        )
                    ),
                    "pretrained_missing_keys": int(
                        student_metadata.get(
                            "pretrained_missing_keys",
                            0,
                        )
                    ),
                }
    elif output_root.exists() and any(output_root.iterdir()):
        raise OfficialRGBDistillationRunnerError("pre-existing output root is not allowed")

    if not config.teacher_checkpoint.is_file():
        raise OfficialRGBDistillationRunnerError(
            "teacher checkpoint must be a regular file"
        )
    if not config.student_pretrained.is_file():
        raise OfficialRGBDistillationRunnerError(
            "student pretrained weights must be a regular file"
        )

    output_root.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(config.device)
    _seed_everything(config.seed)

    train_dataset = dataset_factory(
        config.data_root,
        split="train",
    )
    val_dataset = dataset_factory(
        config.data_root,
        split="val",
    )

    if len(train_dataset) <= 0:
        raise OfficialRGBDistillationRunnerError(
            "training dataset must not be empty"
        )
    if len(val_dataset) <= 0:
        raise OfficialRGBDistillationRunnerError(
            "validation dataset must not be empty"
        )

    train_loader = loader_factory(
        train_dataset,
        batch_size=config.batch,
        shuffle=True,
        num_workers=config.workers,
        collate_fn=collate_paired_batch,
    )
    val_loader = loader_factory(
        val_dataset,
        batch_size=config.batch,
        shuffle=False,
        num_workers=config.workers,
        collate_fn=collate_paired_batch,
    )

    (
        resolved_teacher_factory,
        resolved_teacher_checkpoint_loader,
    ) = _configured_teacher_runtime(
        config,
        teacher_factory=teacher_factory,
        teacher_checkpoint_loader=(teacher_checkpoint_loader),
    )

    teacher = resolved_teacher_factory(
        model_name=config.teacher_model,
        num_classes=config.num_classes,
        device=device,
    )
    teacher_metadata = resolved_teacher_checkpoint_loader(
        teacher,
        config.teacher_checkpoint,
        map_location=device,
    )

    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    # Teacher architectures consume different amounts of randomness.
    # Fresh runs must initialize the student independently of teacher kind.
    if not config.resume:
        _seed_everything(config.seed)

    student = student_factory(
        model_name=config.student_model,
        num_classes=config.num_classes,
        device=device,
    )
    detection_criterion = detection_criterion_factory(student)
    distillation_criterion = _build_distillation_criterion(
        config,
        distillation_criterion_factory,
    )

    feature_distiller = nn.Identity()
    optimizer = optimizer_factory(
        student.parameters(),
        config.lr,
    )

    if config.resume:
        restored = checkpoint_loader(
            last_weights,
            student=student,
            feature_distiller=feature_distiller,
            optimizer=optimizer,
            map_location=device,
        )
        if restored.get("epoch") != completed_before:
            raise OfficialRGBDistillationRunnerError("checkpoint epoch does not match run state")

        pretrained_metadata = {
            "loaded_keys": tuple(range(previous_student_metadata["pretrained_loaded_keys"])),
            "exact_keys": tuple(range(previous_student_metadata["pretrained_exact_keys"])),
            "sliced_keys": tuple(range(previous_student_metadata["pretrained_sliced_keys"])),
            "skipped_classification_keys": tuple(
                range(previous_student_metadata["pretrained_skipped_classification_keys"])
            ),
            "missing_keys": tuple(range(previous_student_metadata["pretrained_missing_keys"])),
            "unexpected_keys": (),
        }
    else:
        pretrained_metadata = (
            _load_configured_student_pretrained(
                config,
                student,
                exact_loader=student_pretrained_loader,
                width_reduced_loader=(
                    student_width_reduced_pretrained_loader
                ),
            )
        )

    completed_epochs: list[int] = []
    final_train_metrics: dict[str, float] = {}
    final_val_metrics: dict[str, float] = {}

    for epoch in range(
        completed_before + 1,
        config.epochs + 1,
    ):
        train_metrics = train_epoch_fn(
            teacher,
            student,
            detection_criterion,
            distillation_criterion,
            optimizer,
            train_loader,
            device=device,
            num_classes=config.num_classes,
        )
        val_metrics = validate_epoch_fn(
            student,
            detection_criterion,
            val_loader,
            device=device,
        )

        val_loss = float(val_metrics["total"])
        checkpoint_metrics = {
            **{
                key: float(value)
                for key, value in train_metrics.items()
            },
            "val_loss": val_loss,
            "val_box": float(val_metrics["box"]),
            "val_classification": float(
                val_metrics["classification"]
            ),
            "val_dfl": float(val_metrics["dfl"]),
        }

        epoch_checkpoint = output_root / "epochs" / f"epoch-{epoch:03d}.pt"
        epoch_checkpoint.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint_saver(
            epoch_checkpoint,
            epoch=epoch,
            student=student,
            feature_distiller=feature_distiller,
            optimizer=optimizer,
            metrics=checkpoint_metrics,
        )
        epoch_weights.append(epoch_checkpoint)

        checkpoint_saver(
            last_weights,
            epoch=epoch,
            student=student,
            feature_distiller=feature_distiller,
            optimizer=optimizer,
            metrics=checkpoint_metrics,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_saver(
                best_weights,
                epoch=epoch,
                student=student,
                feature_distiller=feature_distiller,
                optimizer=optimizer,
                metrics=checkpoint_metrics,
            )

        completed_epochs.append(epoch)
        final_train_metrics = {
            key: float(value)
            for key, value in train_metrics.items()
        }
        final_val_metrics = {
            key: float(value)
            for key, value in val_metrics.items()
        }

        _write_json(
            state_path,
            {
                "format_version": 1,
                "experiment_id": config.experiment_id,
                "status": "running",
                "completed_epoch": epoch,
                "best_val_loss": best_val_loss,
                "last_checkpoint": str(last_weights),
                "best_checkpoint": str(best_weights),
                "epoch_checkpoint": str(epoch_checkpoint),
            },
        )

    _write_json(
        state_path,
        {
            "format_version": 1,
            "experiment_id": config.experiment_id,
            "status": "completed",
            "completed_epoch": completed_epochs[-1],
            "best_val_loss": best_val_loss,
            "last_checkpoint": str(last_weights),
            "best_checkpoint": str(best_weights),
            "epoch_checkpoint": str(epoch_weights[-1]),
        },
    )

    teacher_epoch = int(teacher_metadata.get("epoch", -1))
    loaded_keys = pretrained_metadata.get(
        "loaded_keys",
        (),
    )
    exact_keys = pretrained_metadata.get(
        "exact_keys",
        (),
    )
    sliced_keys = pretrained_metadata.get(
        "sliced_keys",
        (),
    )
    skipped_classification_keys = pretrained_metadata.get(
        "skipped_classification_keys",
        (),
    )
    missing_keys = pretrained_metadata.get(
        "missing_keys",
        (),
    )

    _write_json(
        completion_path,
        {
            "status": "completed",
            "experiment_id": config.experiment_id,
            "experiment_group": config.experiment_group,
            "config": config.to_metadata(),
            "dataset": {
                "train_count": len(train_dataset),
                "val_count": len(val_dataset),
            },
            "teacher": {
                "kind": config.teacher_kind,
                "model": config.teacher_model,
                "checkpoint": str(config.teacher_checkpoint),
                "restored_epoch": teacher_epoch,
            },
                        "student": {
                "model": config.student_model,
                "pretrained": str(
                    config.student_pretrained
                ),
                "pretrained_kind": (
                    config.student_pretrained_kind
                ),
                "pretrained_loaded_keys": len(
                    loaded_keys
                ),
                "pretrained_exact_keys": len(
                    exact_keys
                ),
                "pretrained_sliced_keys": len(
                    sliced_keys
                ),
                "pretrained_skipped_classification_keys": len(
                    skipped_classification_keys
                ),
                "pretrained_missing_keys": len(
                    missing_keys
                ),
            },
            "completed_epochs": completed_epochs,
            "all_completed_epochs": list(range(1, config.epochs + 1)),
            "best_weights": str(best_weights),
            "last_weights": str(last_weights),
            "state_path": str(state_path),
            "metrics": {
                "val_loss": best_val_loss,
            },
            "final_train_metrics": final_train_metrics,
            "final_val_metrics": final_val_metrics,
            "epoch_weights": [str(path) for path in epoch_weights],
        },
    )

    return OfficialRGBDistillationResult(
        best_weights=best_weights,
        last_weights=last_weights,
        state_path=state_path,
        completion_path=completion_path,
        completed_epochs=tuple(completed_epochs),
        metrics={
            "val_loss": best_val_loss,
        },
    )