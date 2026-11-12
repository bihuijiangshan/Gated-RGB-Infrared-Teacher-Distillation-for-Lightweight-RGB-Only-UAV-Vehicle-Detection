"""Build a deterministic formal unimodal training plan."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rgbt_uav.training.formal_unimodal_config import (
    FormalUnimodalConfig,
)


class FormalUnimodalTrainingPlanError(
    ValueError
):
    """Raised when a formal unimodal plan cannot be built."""


@dataclass(frozen=True)
class FormalUnimodalTrainingPlan:
    """Immutable paths and arguments for one formal training run."""

    experiment_id: str
    modality: str
    model_source: str
    train_arguments: dict[str, object]
    training_root: Path
    best_weights: Path
    last_weights: Path
    completion_path: Path
    starting_epoch: int


def _portable_path(
    path: Path,
) -> str:
    return str(path).replace(
        "\\",
        "/",
    )


def build_formal_unimodal_training_plan(
    config: FormalUnimodalConfig,
    *,
    resume_checkpoint: Path | None = None,
) -> FormalUnimodalTrainingPlan:
    """Translate a validated config into exact training arguments."""

    if not isinstance(
        config,
        FormalUnimodalConfig,
    ):
        raise FormalUnimodalTrainingPlanError(
            "config must be a FormalUnimodalConfig"
        )

    training_root = (
        config.output_root / "train"
    )
    weights_root = (
        training_root / "weights"
    )

    train_arguments: dict[
        str,
        object,
    ] = {
        "data": _portable_path(
            config.data
        ),
        "epochs": config.epochs,
        "imgsz": config.imgsz,
        "rect": config.rect,
        "batch": config.batch,
        "seed": config.seed,
        "workers": config.workers,
        "device": config.device,
        "amp": config.amp,
        "cache": config.cache,
        "optimizer": config.optimizer,
        "lr0": float(config.lr),
        "weight_decay": float(
            config.weight_decay
        ),
        "project": _portable_path(
            config.output_root
        ),
        "name": "train",
        "exist_ok": False,
        "save": True,
        "save_period": -1,
        "val": True,
        "plots": True,
    }

    model_source = config.model
    if resume_checkpoint is not None:
        checkpoint = Path(
            resume_checkpoint
        )
        model_source = _portable_path(
            checkpoint
        )
        train_arguments["resume"] = True
        train_arguments["exist_ok"] = True

    serialized = repr(
        train_arguments
    ).replace(
        "\\",
        "/",
    ).lower()

    if (
        "/images/test" in serialized
        or "/labels/test" in serialized
        or "split=test" in serialized
        or "'split': 'test'" in serialized
    ):
        raise FormalUnimodalTrainingPlanError(
            "training plan must not reference the test split"
        )

    return FormalUnimodalTrainingPlan(
        experiment_id=(
            config.experiment_id
        ),
        modality=config.modality,
        model_source=model_source,
        train_arguments=train_arguments,
        training_root=training_root,
        best_weights=(
            weights_root / "best.pt"
        ),
        last_weights=(
            weights_root / "last.pt"
        ),
        completion_path=(
            config.output_root
            / "completion.json"
        ),
        starting_epoch=(
            0
            if resume_checkpoint is None
            else _completed_epoch_from_checkpoint(
                Path(resume_checkpoint)
            )
        ),
    )


def _completed_epoch_from_checkpoint(
    checkpoint: Path,
) -> int:
    epoch_root = checkpoint.parent.parent / "epochs"
    retained: list[int] = []

    if epoch_root.is_dir() and not epoch_root.is_symlink():
        for path in epoch_root.iterdir():
            if not path.is_file() or path.is_symlink():
                continue
            name = path.name
            if (
                name.startswith("epoch-")
                and name.endswith(".pt")
                and name[6:-3].isdigit()
            ):
                retained.append(int(name[6:-3]))

    if not retained:
        raise FormalUnimodalTrainingPlanError(
            "resume checkpoint has no retained epoch history"
        )

    completed_epoch = max(retained)
    if sorted(retained) != list(range(1, completed_epoch + 1)):
        raise FormalUnimodalTrainingPlanError(
            "resume checkpoint history must be sequential"
        )

    return completed_epoch
