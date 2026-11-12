"""Ultralytics backend for formal unimodal training."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rgbt_uav.training.formal_epoch_checkpoint import (
    FormalEpochCheckpointRetainer,
)
from rgbt_uav.training.formal_unimodal_plan import (
    FormalUnimodalTrainingPlan,
)


class FormalUnimodalBackendError(
    RuntimeError
):
    """Raised when the Ultralytics backend violates its contract."""


@dataclass(frozen=True)
class FormalUnimodalBackendResult:
    """Checkpoint history produced by the training backend."""

    retained_epochs: tuple[int, ...]
    retained_paths: tuple[Path, ...]


def _default_model_factory(
    model_source: str,
) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise FormalUnimodalBackendError(
            "ultralytics is required for formal training"
        ) from exc

    try:
        return YOLO(model_source)
    except Exception as exc:
        raise FormalUnimodalBackendError(
            "failed to construct the Ultralytics model"
        ) from exc


def run_ultralytics_formal_training(
    plan: FormalUnimodalTrainingPlan,
    *,
    model_factory: Callable[
        [str],
        Any,
    ] = _default_model_factory,
) -> FormalUnimodalBackendResult:
    """Train once and retain every post-save epoch checkpoint."""

    if not isinstance(
        plan,
        FormalUnimodalTrainingPlan,
    ):
        raise FormalUnimodalBackendError(
            "plan must be a FormalUnimodalTrainingPlan"
        )

    if not callable(model_factory):
        raise FormalUnimodalBackendError(
            "model_factory must be callable"
        )

    raw_epochs = plan.train_arguments.get(
        "epochs"
    )
    if (
        isinstance(raw_epochs, bool)
        or not isinstance(
            raw_epochs,
            int,
        )
        or raw_epochs <= 0
    ):
        raise FormalUnimodalBackendError(
            "training plan epochs must be positive"
        )

    if (
        isinstance(plan.starting_epoch, bool)
        or not isinstance(plan.starting_epoch, int)
        or plan.starting_epoch < 0
        or plan.starting_epoch >= raw_epochs
    ):
        raise FormalUnimodalBackendError(
            "starting_epoch must be smaller than training epochs"
        )

    if (
        plan.train_arguments.get(
            "save_period"
        )
        != -1
    ):
        raise FormalUnimodalBackendError(
            "native periodic checkpoints must be disabled"
        )

    try:
        model = model_factory(
            plan.model_source
        )
    except FormalUnimodalBackendError:
        raise
    except Exception as exc:
        raise FormalUnimodalBackendError(
            "model_factory failed"
        ) from exc

    add_callback = getattr(
        model,
        "add_callback",
        None,
    )
    if not callable(add_callback):
        raise FormalUnimodalBackendError(
            "model must provide add_callback"
        )

    train = getattr(
        model,
        "train",
        None,
    )
    if not callable(train):
        raise FormalUnimodalBackendError(
            "model must provide train"
        )

    retainer = (
        FormalEpochCheckpointRetainer(
            plan.training_root
            / "epochs",
            expected_epochs=raw_epochs,
            starting_epoch=plan.starting_epoch,
        )
    )

    try:
        add_callback(
            "on_model_save",
            retainer,
        )
    except Exception as exc:
        raise FormalUnimodalBackendError(
            "failed to register on_model_save callback"
        ) from exc

    try:
        train(
            **plan.train_arguments
        )
    except Exception as exc:
        raise FormalUnimodalBackendError(
            "Ultralytics formal training failed"
        ) from exc

    expected_epochs = tuple(
        range(
            1,
            raw_epochs + 1,
        )
    )

    if (
        retainer.retained_epochs
        != expected_epochs
        or len(
            retainer.retained_paths
        )
        != raw_epochs
    ):
        raise FormalUnimodalBackendError(
            "checkpoint history is incomplete"
        )

    for path in retainer.retained_paths:
        if (
            not path.is_file()
            or path.is_symlink()
        ):
            raise FormalUnimodalBackendError(
                "checkpoint history contains "
                f"an invalid artifact: {path}"
            )

    return FormalUnimodalBackendResult(
        retained_epochs=(
            retainer.retained_epochs
        ),
        retained_paths=(
            retainer.retained_paths
        ),
    )
