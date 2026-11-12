"""Checkpoint persistence for the official gated teacher."""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch.optim import Optimizer

from rgbt_uav.models.official_gated_teacher import (
    OfficialGatedDualStreamTeacher,
)


class OfficialGatedTeacherCheckpointError(
    RuntimeError
):
    """Raised when a gated-teacher checkpoint is invalid."""


_FORMAT_VERSION = 1

_REQUIRED_FIELDS = {
    "format_version",
    "epoch",
    "experiment_id",
    "model_name",
    "pretrained",
    "teacher",
    "optimizer",
    "metrics",
}


def _non_empty_string(
    value: object,
    *,
    field: str,
) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise OfficialGatedTeacherCheckpointError(
            f"{field} must be a non-empty string"
        )

    return value.strip()


def _validated_epoch(
    value: object,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise OfficialGatedTeacherCheckpointError(
            "epoch must be a positive integer"
        )

    return value


def _validated_metrics(
    metrics: object,
) -> dict[str, float]:
    if not isinstance(metrics, Mapping):
        raise OfficialGatedTeacherCheckpointError(
            "metrics must be a mapping"
        )

    result: dict[str, float] = {}

    for key, value in metrics.items():
        if (
            not isinstance(key, str)
            or not key
        ):
            raise OfficialGatedTeacherCheckpointError(
                "metric names must be "
                "non-empty strings"
            )

        if (
            isinstance(value, bool)
            or not isinstance(
                value,
                (int, float),
            )
        ):
            raise OfficialGatedTeacherCheckpointError(
                f"metric {key!r} must "
                "be numeric"
            )

        number = float(value)

        if not math.isfinite(number):
            raise OfficialGatedTeacherCheckpointError(
                f"metric {key!r} must "
                "be finite"
            )

        result[key] = number

    return result


def _mapping(
    value: object,
    *,
    field: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OfficialGatedTeacherCheckpointError(
            f"checkpoint field {field!r} "
            "must be a mapping"
        )

    return value


def save_official_gated_teacher_checkpoint(
    path: Path,
    *,
    teacher: OfficialGatedDualStreamTeacher,
    optimizer: Optimizer,
    epoch: int,
    metrics: Mapping[str, object],
    experiment_id: str,
    model_name: str,
    pretrained: Path,
) -> Path:
    """Atomically save complete gated-teacher training state."""

    if not isinstance(
        teacher,
        OfficialGatedDualStreamTeacher,
    ):
        raise OfficialGatedTeacherCheckpointError(
            "teacher must be an "
            "OfficialGatedDualStreamTeacher"
        )

    if not isinstance(optimizer, Optimizer):
        raise OfficialGatedTeacherCheckpointError(
            "optimizer must be a torch optimizer"
        )

    validated_epoch = _validated_epoch(
        epoch
    )
    validated_experiment_id = (
        _non_empty_string(
            experiment_id,
            field="experiment_id",
        )
    )
    validated_model_name = (
        _non_empty_string(
            model_name,
            field="model_name",
        )
    )

    pretrained_text = str(
        Path(pretrained)
    )

    if not pretrained_text.strip():
        raise OfficialGatedTeacherCheckpointError(
            "pretrained must identify "
            "the initialization source"
        )

    validated_metrics = (
        _validated_metrics(metrics)
    )

    destination = Path(path)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "format_version": _FORMAT_VERSION,
        "epoch": validated_epoch,
        "experiment_id": (
            validated_experiment_id
        ),
        "model_name": validated_model_name,
        "pretrained": pretrained_text,
        "teacher": teacher.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": validated_metrics,
    }

    temporary_path: Path | None = None

    try:
        temporary = (
            tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=".",
                suffix=".gated-teacher.tmp",
                dir=destination.parent,
                delete=False,
            )
        )
        temporary_path = Path(
            temporary.name
        )
        temporary.close()

        torch.save(
            payload,
            temporary_path,
        )
        os.replace(
            temporary_path,
            destination,
        )
        temporary_path = None
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise OfficialGatedTeacherCheckpointError(
            "failed to save gated-teacher "
            f"checkpoint: {destination}"
        ) from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(
                    missing_ok=True
                )
            except OSError:
                pass

    return destination


def load_official_gated_teacher_checkpoint(
    path: Path,
    *,
    teacher: OfficialGatedDualStreamTeacher,
    optimizer: Optimizer,
    expected_experiment_id: str,
    map_location: torch.device | str,
) -> dict[str, object]:
    """Restore complete gated-teacher training state."""

    if not isinstance(
        teacher,
        OfficialGatedDualStreamTeacher,
    ):
        raise OfficialGatedTeacherCheckpointError(
            "teacher must be an "
            "OfficialGatedDualStreamTeacher"
        )

    if not isinstance(optimizer, Optimizer):
        raise OfficialGatedTeacherCheckpointError(
            "optimizer must be a torch optimizer"
        )

    expected_id = _non_empty_string(
        expected_experiment_id,
        field="expected_experiment_id",
    )

    source = Path(path)

    if not source.is_file():
        raise OfficialGatedTeacherCheckpointError(
            "checkpoint must be a regular "
            f"file: {source}"
        )

    try:
        payload: Any = torch.load(
            source,
            map_location=map_location,
            weights_only=False,
        )
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise OfficialGatedTeacherCheckpointError(
            "failed to load gated-teacher "
            f"checkpoint: {source}"
        ) from exc

    checkpoint = _mapping(
        payload,
        field="root",
    )

    missing_fields = (
        _REQUIRED_FIELDS
        - set(checkpoint)
    )

    if missing_fields:
        names = ", ".join(
            sorted(missing_fields)
        )
        raise OfficialGatedTeacherCheckpointError(
            "checkpoint is missing fields: "
            f"{names}"
        )

    if (
        checkpoint["format_version"]
        != _FORMAT_VERSION
    ):
        raise OfficialGatedTeacherCheckpointError(
            "unsupported checkpoint "
            "format version"
        )

    epoch = _validated_epoch(
        checkpoint["epoch"]
    )
    experiment_id = _non_empty_string(
        checkpoint["experiment_id"],
        field="checkpoint experiment_id",
    )

    if experiment_id != expected_id:
        raise OfficialGatedTeacherCheckpointError(
            "checkpoint experiment_id "
            "does not match the requested experiment"
        )

    model_name = _non_empty_string(
        checkpoint["model_name"],
        field="checkpoint model_name",
    )
    pretrained = _non_empty_string(
        checkpoint["pretrained"],
        field="checkpoint pretrained",
    )
    metrics = _validated_metrics(
        checkpoint["metrics"]
    )

    teacher_state = _mapping(
        checkpoint["teacher"],
        field="teacher",
    )
    optimizer_state = _mapping(
        checkpoint["optimizer"],
        field="optimizer",
    )

    try:
        teacher.load_state_dict(
            teacher_state,
            strict=True,
        )
        optimizer.load_state_dict(
            optimizer_state
        )
    except (
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise OfficialGatedTeacherCheckpointError(
            "checkpoint state is incompatible "
            "with teacher or optimizer"
        ) from exc

    return {
        "format_version": _FORMAT_VERSION,
        "epoch": epoch,
        "experiment_id": experiment_id,
        "model_name": model_name,
        "pretrained": pretrained,
        "metrics": metrics,
    }

def load_official_gated_teacher_weights(
    path: Path,
    *,
    teacher: OfficialGatedDualStreamTeacher,
    map_location: torch.device | str,
) -> dict[str, object]:
    """Load and freeze gated-teacher weights without optimizer state."""

    if not isinstance(
        teacher,
        OfficialGatedDualStreamTeacher,
    ):
        raise OfficialGatedTeacherCheckpointError(
            "teacher must be an "
            "OfficialGatedDualStreamTeacher"
        )

    source = Path(path)

    if not source.is_file():
        raise OfficialGatedTeacherCheckpointError(
            "checkpoint must be a regular "
            f"file: {source}"
        )

    try:
        payload: Any = torch.load(
            source,
            map_location=map_location,
            weights_only=False,
        )
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise OfficialGatedTeacherCheckpointError(
            "failed to load gated-teacher "
            f"checkpoint: {source}"
        ) from exc

    checkpoint = _mapping(
        payload,
        field="root",
    )

    missing_fields = (
        _REQUIRED_FIELDS
        - set(checkpoint)
    )

    if missing_fields:
        names = ", ".join(
            sorted(missing_fields)
        )
        raise OfficialGatedTeacherCheckpointError(
            "checkpoint is missing fields: "
            f"{names}"
        )

    if (
        checkpoint["format_version"]
        != _FORMAT_VERSION
    ):
        raise OfficialGatedTeacherCheckpointError(
            "unsupported checkpoint "
            "format version"
        )

    epoch = _validated_epoch(
        checkpoint["epoch"]
    )
    experiment_id = _non_empty_string(
        checkpoint["experiment_id"],
        field="checkpoint experiment_id",
    )
    model_name = _non_empty_string(
        checkpoint["model_name"],
        field="checkpoint model_name",
    )
    pretrained = _non_empty_string(
        checkpoint["pretrained"],
        field="checkpoint pretrained",
    )
    metrics = _validated_metrics(
        checkpoint["metrics"]
    )

    teacher_state = _mapping(
        checkpoint["teacher"],
        field="teacher",
    )

    try:
        teacher.load_state_dict(
            teacher_state,
            strict=True,
        )
    except (
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise OfficialGatedTeacherCheckpointError(
            "checkpoint state is incompatible "
            "with the gated teacher"
        ) from exc

    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    teacher.eval()

    return {
        "format_version": _FORMAT_VERSION,
        "epoch": epoch,
        "experiment_id": experiment_id,
        "model_name": model_name,
        "pretrained": pretrained,
        "metrics": metrics,
    }