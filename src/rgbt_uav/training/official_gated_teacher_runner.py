"""Fresh-run orchestration for the official gated teacher."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from rgbt_uav.data.paired import (
    PairedYoloDataset,
    collate_paired_batch,
)
from rgbt_uav.models.official_gated_teacher import (
    build_official_gated_teacher,
)
from rgbt_uav.training.official_gated_teacher_checkpoint import (
    load_official_gated_teacher_checkpoint,
    save_official_gated_teacher_checkpoint,
)
from rgbt_uav.training.official_gated_teacher_config import (
    OfficialGatedTeacherConfig,
)
from rgbt_uav.training.official_gated_teacher_epoch import (
    train_official_gated_teacher_epoch,
    validate_official_gated_teacher_epoch,
)
from rgbt_uav.training.official_gated_teacher_loss import (
    OfficialGatedTeacherDetectionLoss,
)
from rgbt_uav.training.official_gated_teacher_pretrained import (
    load_official_gated_teacher_pretrained,
)


class OfficialGatedTeacherRunnerError(
    RuntimeError
):
    """Raised when gated-teacher orchestration fails."""


@dataclass(frozen=True)
class OfficialGatedTeacherResult:
    """Artifacts produced by a gated-teacher run."""

    best_weights: Path
    last_weights: Path
    state_path: Path
    completion_path: Path
    completed_epochs: tuple[int, ...]
    metrics: dict[str, float]


@dataclass(frozen=True)
class _ResumeEvidence:
    completed_epoch: int
    best_val_loss: float
    epoch_weights: tuple[Path, ...]
    pretrained: Mapping[str, object]


def _resolve_device(
    value: str,
) -> torch.device:
    normalized = value.strip().lower()

    if normalized == "cpu":
        return torch.device("cpu")

    if normalized.isdigit():
        device = torch.device(
            f"cuda:{normalized}"
        )
    elif normalized.startswith("cuda"):
        device = torch.device(
            normalized
        )
    else:
        raise OfficialGatedTeacherRunnerError(
            f"unsupported device: {value}"
        )

    if not torch.cuda.is_available():
        raise OfficialGatedTeacherRunnerError(
            "CUDA was requested but "
            "is unavailable"
        )

    return device


def _seed_everything(
    seed: int,
) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_json(
    path: Path,
    payload: Mapping[str, object],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    try:
        temporary.write_text(
            json.dumps(
                dict(payload),
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    except (
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        try:
            temporary.unlink(
                missing_ok=True
            )
        except OSError:
            pass

        raise OfficialGatedTeacherRunnerError(
            "failed to write JSON evidence: "
            f"{destination}"
        ) from exc

    return destination

def _read_json(
    path: Path,
    *,
    label: str,
) -> Mapping[str, object]:
    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as exc:
        raise OfficialGatedTeacherRunnerError(
            f"failed to read {label}: {path}"
        ) from exc

    if not isinstance(payload, Mapping):
        raise OfficialGatedTeacherRunnerError(
            f"{label} must be a mapping"
        )

    return payload


def _optimizer_factory(
    parameters: Any,
    *,
    lr: float,
    weight_decay: float,
) -> AdamW:
    return AdamW(
        parameters,
        lr=lr,
        weight_decay=weight_decay,
    )


def _finite_metrics(
    value: object,
    *,
    label: str,
) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise OfficialGatedTeacherRunnerError(
            f"{label} must be a mapping"
        )

    result: dict[str, float] = {}

    for key, item in value.items():
        if (
            not isinstance(key, str)
            or not key
        ):
            raise OfficialGatedTeacherRunnerError(
                f"{label} contains an "
                "invalid metric name"
            )

        if (
            isinstance(item, bool)
            or not isinstance(
                item,
                (int, float),
            )
        ):
            raise OfficialGatedTeacherRunnerError(
                f"{label}.{key} must "
                "be numeric"
            )

        number = float(item)

        if not math.isfinite(number):
            raise OfficialGatedTeacherRunnerError(
                f"{label}.{key} must "
                "be finite"
            )

        result[key] = number

    return result


def _metadata_count(
    metadata: Mapping[str, object],
    key: str,
) -> int:
    value = metadata.get(
        key,
        (),
    )

    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    ):
        return value

    if isinstance(
        value,
        (list, tuple),
    ):
        return len(value)

    raise OfficialGatedTeacherRunnerError(
        "pretrained metadata field "
        f"{key!r} must be a sequence "
        "or non-negative integer"
    )


def _load_resume_evidence(
    config: OfficialGatedTeacherConfig,
    *,
    state_path: Path,
    completion_path: Path,
    last_weights: Path,
    best_weights: Path,
) -> _ResumeEvidence:
    for path in (
        state_path,
        completion_path,
        last_weights,
        best_weights,
    ):
        if not path.is_file():
            raise OfficialGatedTeacherRunnerError(
                "resume artifact must be "
                f"a regular file: {path}"
            )

    state = _read_json(
        state_path,
        label="run state",
    )
    completion = _read_json(
        completion_path,
        label="completion evidence",
    )

    if state.get("format_version") != 1:
        raise OfficialGatedTeacherRunnerError(
            "unsupported run-state "
            "format version"
        )

    if (
        state.get("experiment_id")
        != config.experiment_id
    ):
        raise OfficialGatedTeacherRunnerError(
            "resume experiment_id "
            "does not match"
        )

    if (
        completion.get("experiment_id")
        != config.experiment_id
    ):
        raise OfficialGatedTeacherRunnerError(
            "completion experiment_id "
            "does not match"
        )

    completed_epoch = state.get(
        "completed_epoch"
    )

    if (
        isinstance(completed_epoch, bool)
        or not isinstance(
            completed_epoch,
            int,
        )
        or completed_epoch <= 0
    ):
        raise OfficialGatedTeacherRunnerError(
            "resume completed_epoch "
            "must be positive"
        )

    if completed_epoch >= config.epochs:
        raise OfficialGatedTeacherRunnerError(
            "resume target must exceed "
            "completed epoch"
        )

    best_value = state.get(
        "best_val_loss"
    )

    if (
        isinstance(best_value, bool)
        or not isinstance(
            best_value,
            (int, float),
        )
        or not math.isfinite(
            float(best_value)
        )
    ):
        raise OfficialGatedTeacherRunnerError(
            "resume best_val_loss "
            "must be finite"
        )

    previous_config = completion.get(
        "config"
    )

    if not isinstance(
        previous_config,
        Mapping,
    ):
        raise OfficialGatedTeacherRunnerError(
            "resume completion config "
            "must be a mapping"
        )

    current_config = (
        config.to_metadata()
    )

    stable_fields = (
        "model",
        "pretrained",
        "batch",
        "imgsz",
        "seed",
        "device",
        "workers",
        "lr",
        "weight_decay",
        "num_classes",
    )

    for field in stable_fields:
        if (
            previous_config.get(field)
            != current_config[field]
        ):
            raise OfficialGatedTeacherRunnerError(
                "resume configuration "
                f"field {field!r} does not match"
            )

    previous_all_epochs = completion.get(
        "all_completed_epochs"
    )

    expected_epochs = list(
        range(
            1,
            completed_epoch + 1,
        )
    )

    if previous_all_epochs != expected_epochs:
        raise OfficialGatedTeacherRunnerError(
            "resume completed epoch "
            "history is inconsistent"
        )

    raw_epoch_weights = completion.get(
        "epoch_weights"
    )

    if not isinstance(
        raw_epoch_weights,
        list,
    ):
        raise OfficialGatedTeacherRunnerError(
            "resume epoch_weights "
            "must be a list"
        )

    epoch_weights: list[Path] = []

    for value in raw_epoch_weights:
        if (
            not isinstance(value, str)
            or not value
        ):
            raise OfficialGatedTeacherRunnerError(
                "resume epoch checkpoint "
                "path is invalid"
            )

        checkpoint = Path(value)

        if not checkpoint.is_file():
            raise OfficialGatedTeacherRunnerError(
                "resume epoch checkpoint "
                f"is missing: {checkpoint}"
            )

        epoch_weights.append(
            checkpoint
        )

    if len(epoch_weights) != completed_epoch:
        raise OfficialGatedTeacherRunnerError(
            "resume epoch checkpoint "
            "history length is inconsistent"
        )

    pretrained = completion.get(
        "pretrained"
    )

    if not isinstance(
        pretrained,
        Mapping,
    ):
        raise OfficialGatedTeacherRunnerError(
            "resume pretrained metadata "
            "must be a mapping"
        )

    if pretrained.get("path") != str(
        config.pretrained
    ):
        raise OfficialGatedTeacherRunnerError(
            "resume pretrained source "
            "does not match"
        )

    for key in (
        "rgb_loaded_keys",
        "rgb_missing_keys",
        "ir_loaded_keys",
        "ir_adapted_keys",
        "ir_missing_keys",
        "detect_loaded_keys",
        "detect_missing_keys",
    ):
        _metadata_count(
            pretrained,
            key,
        )

    return _ResumeEvidence(
        completed_epoch=completed_epoch,
        best_val_loss=float(
            best_value
        ),
        epoch_weights=tuple(
            epoch_weights
        ),
        pretrained=pretrained,
    )


def _checkpoint_metrics(
    train_metrics: Mapping[str, float],
    val_metrics: Mapping[str, float],
) -> dict[str, float]:
    if "total" not in val_metrics:
        raise OfficialGatedTeacherRunnerError(
            "validation metrics are "
            "missing total"
        )

    result = {
        key: float(value)
        for key, value in (
            train_metrics.items()
        )
    }
    result["val_loss"] = float(
        val_metrics["total"]
    )

    for key, value in val_metrics.items():
        if key == "total":
            continue

        result[f"val_{key}"] = float(
            value
        )

    return result


def run_official_gated_teacher(
    config: OfficialGatedTeacherConfig,
    *,
    dataset_factory: Callable[
        ...,
        Any,
    ] = PairedYoloDataset,
    loader_factory: Callable[
        ...,
        Any,
    ] = DataLoader,
    teacher_factory: Callable[
        ...,
        Any,
    ] = build_official_gated_teacher,
    pretrained_loader: Callable[
        ...,
        Any,
    ] = load_official_gated_teacher_pretrained,
    criterion_factory: Callable[
        ...,
        Any,
    ] = OfficialGatedTeacherDetectionLoss,
    optimizer_factory: Callable[
        ...,
        Any,
    ] = _optimizer_factory,
    train_epoch_fn: Callable[
        ...,
        Any,
    ] = train_official_gated_teacher_epoch,
    validate_epoch_fn: Callable[
        ...,
        Any,
    ] = validate_official_gated_teacher_epoch,
    checkpoint_saver: Callable[
        ...,
        Any,
    ] = save_official_gated_teacher_checkpoint,
    checkpoint_loader: Callable[
        ...,
        Any,
    ] = load_official_gated_teacher_checkpoint,
) -> OfficialGatedTeacherResult:
    """Run a fresh official gated-teacher experiment."""

    if not isinstance(
        config,
        OfficialGatedTeacherConfig,
    ):
        raise OfficialGatedTeacherRunnerError(
            "config must be an "
            "OfficialGatedTeacherConfig"
        )

    output_root = config.output_root
    best_weights = (
        output_root / "best.pt"
    )
    last_weights = (
        output_root / "last.pt"
    )
    state_path = (
        output_root / "run_state.json"
    )
    completion_path = (
        output_root / "completion.json"
    )

    completed_before = 0
    best_val_loss = float("inf")
    epoch_weights: list[Path] = []
    pretrained_metadata: Mapping[
        str,
        object,
    ]

    if config.resume:
        evidence = _load_resume_evidence(
            config,
            state_path=state_path,
            completion_path=completion_path,
            last_weights=last_weights,
            best_weights=best_weights,
        )
        completed_before = evidence.completed_epoch
        best_val_loss = evidence.best_val_loss
        epoch_weights.extend(evidence.epoch_weights)
        pretrained_metadata = evidence.pretrained
    elif output_root.exists() and any(output_root.iterdir()):
        raise OfficialGatedTeacherRunnerError("pre-existing output root is not allowed")

    if not config.pretrained.is_file():
        raise OfficialGatedTeacherRunnerError(
            "pretrained weights must "
            "be a regular file"
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = _resolve_device(
        config.device
    )
    _seed_everything(
        config.seed
    )

    try:
        train_dataset = dataset_factory(
            config.data_root,
            split="train",
        )
        val_dataset = dataset_factory(
            config.data_root,
            split="val",
        )
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise OfficialGatedTeacherRunnerError(
            "failed to build paired datasets"
        ) from exc

    if len(train_dataset) <= 0:
        raise OfficialGatedTeacherRunnerError(
            "training dataset must not "
            "be empty"
        )

    if len(val_dataset) <= 0:
        raise OfficialGatedTeacherRunnerError(
            "validation dataset must not "
            "be empty"
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

    try:
        teacher = teacher_factory(
            model_name=config.model,
            num_classes=config.num_classes,
            device=device,
        )
        criterion = criterion_factory(teacher)
        optimizer = optimizer_factory(
            teacher.parameters(),
            lr=float(config.lr),
            weight_decay=float(config.weight_decay),
        )
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise OfficialGatedTeacherRunnerError(
            "failed to initialize gated teacher training"
        ) from exc

    if config.resume:
        try:
            restored = checkpoint_loader(
                last_weights,
                teacher=teacher,
                optimizer=optimizer,
                expected_experiment_id=(config.experiment_id),
                map_location=device,
            )
        except (
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            raise OfficialGatedTeacherRunnerError(
                "failed to restore gated-teacher checkpoint"
            ) from exc

        if not isinstance(
            restored,
            Mapping,
        ):
            raise OfficialGatedTeacherRunnerError("checkpoint loader must return a mapping")

        if restored.get("epoch") != completed_before:
            raise OfficialGatedTeacherRunnerError("checkpoint epoch does not match run state")

        if restored.get("experiment_id") != config.experiment_id:
            raise OfficialGatedTeacherRunnerError("checkpoint experiment_id does not match")

        if restored.get("model_name") != config.model:
            raise OfficialGatedTeacherRunnerError("checkpoint model_name does not match")

        if restored.get("pretrained") != str(config.pretrained):
            raise OfficialGatedTeacherRunnerError("checkpoint pretrained source does not match")
    else:
        try:
            loaded_metadata = pretrained_loader(
                teacher,
                config.pretrained,
            )
        except (
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            raise OfficialGatedTeacherRunnerError(
                "failed to initialize gated teacher from pretrained weights"
            ) from exc

        if not isinstance(
            loaded_metadata,
            Mapping,
        ):
            raise OfficialGatedTeacherRunnerError("pretrained loader must return a mapping")

        pretrained_metadata = loaded_metadata

    completed_epochs: list[int] = []
    final_train_metrics: dict[
        str,
        float,
    ] = {}
    final_val_metrics: dict[
        str,
        float,
    ] = {}

    for epoch in range(
        completed_before + 1,
        config.epochs + 1,
    ):
        try:
            train_metrics = _finite_metrics(
                train_epoch_fn(
                    teacher,
                    criterion,
                    optimizer,
                    train_loader,
                    device=device,
                ),
                label="training metrics",
            )
            val_metrics = _finite_metrics(
                validate_epoch_fn(
                    teacher,
                    criterion,
                    val_loader,
                    device=device,
                ),
                label="validation metrics",
            )
        except (
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            raise OfficialGatedTeacherRunnerError(
                f"epoch {epoch} failed"
            ) from exc

        metrics = _checkpoint_metrics(
            train_metrics,
            val_metrics,
        )
        val_loss = metrics[
            "val_loss"
        ]

        epoch_checkpoint = (
            output_root
            / "epochs"
            / f"epoch-{epoch:03d}.pt"
        )
        epoch_checkpoint.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint_arguments = {
            "teacher": teacher,
            "optimizer": optimizer,
            "epoch": epoch,
            "metrics": metrics,
            "experiment_id": (
                config.experiment_id
            ),
            "model_name": config.model,
            "pretrained": (
                config.pretrained
            ),
        }

        checkpoint_saver(
            epoch_checkpoint,
            **checkpoint_arguments,
        )
        epoch_weights.append(
            epoch_checkpoint
        )

        checkpoint_saver(
            last_weights,
            **checkpoint_arguments,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_saver(
                best_weights,
                **checkpoint_arguments,
            )

        completed_epochs.append(
            epoch
        )
        final_train_metrics = (
            train_metrics
        )
        final_val_metrics = val_metrics

        _write_json(
            state_path,
            {
                "format_version": 1,
                "experiment_id": (
                    config.experiment_id
                ),
                "status": "running",
                "completed_epoch": epoch,
                "best_val_loss": (
                    best_val_loss
                ),
                "last_checkpoint": str(
                    last_weights
                ),
                "best_checkpoint": str(
                    best_weights
                ),
                "epoch_checkpoint": str(
                    epoch_checkpoint
                ),
            },
        )

    _write_json(
        state_path,
        {
            "format_version": 1,
            "experiment_id": (
                config.experiment_id
            ),
            "status": "completed",
            "completed_epoch": (
                completed_epochs[-1]
            ),
            "best_val_loss": (
                best_val_loss
            ),
            "last_checkpoint": str(
                last_weights
            ),
            "best_checkpoint": str(
                best_weights
            ),
            "epoch_checkpoint": str(
                epoch_weights[-1]
            ),
        },
    )

    completion = {
        "status": "completed",
        "experiment_id": (
            config.experiment_id
        ),
        "experiment_group": (
            config.experiment_group
        ),
        "config": config.to_metadata(),
        "dataset": {
            "train_count": len(
                train_dataset
            ),
            "val_count": len(
                val_dataset
            ),
        },
        "completed_epochs": (
            completed_epochs
        ),
        "all_completed_epochs": list(
            range(
                1,
                config.epochs + 1,
            )
        ),
        "best_weights": str(
            best_weights
        ),
        "last_weights": str(
            last_weights
        ),
        "state_path": str(
            state_path
        ),
        "metrics": {
            "val_loss": best_val_loss,
        },
        "final_train_metrics": (
            final_train_metrics
        ),
        "final_val_metrics": (
            final_val_metrics
        ),
        "epoch_weights": [
            str(path)
            for path in epoch_weights
        ],
        "pretrained": {
            "path": str(
                config.pretrained
            ),
            "rgb_loaded_keys": (
                _metadata_count(
                    pretrained_metadata,
                    "rgb_loaded_keys",
                )
            ),
            "rgb_missing_keys": (
                _metadata_count(
                    pretrained_metadata,
                    "rgb_missing_keys",
                )
            ),
            "ir_loaded_keys": (
                _metadata_count(
                    pretrained_metadata,
                    "ir_loaded_keys",
                )
            ),
            "ir_adapted_keys": (
                _metadata_count(
                    pretrained_metadata,
                    "ir_adapted_keys",
                )
            ),
            "ir_missing_keys": (
                _metadata_count(
                    pretrained_metadata,
                    "ir_missing_keys",
                )
            ),
            "detect_loaded_keys": (
                _metadata_count(
                    pretrained_metadata,
                    "detect_loaded_keys",
                )
            ),
            "detect_missing_keys": (
                _metadata_count(
                    pretrained_metadata,
                    "detect_missing_keys",
                )
            ),
        },
    }

    _write_json(
        completion_path,
        completion,
    )

    return OfficialGatedTeacherResult(
        best_weights=best_weights,
        last_weights=last_weights,
        state_path=state_path,
        completion_path=completion_path,
        completed_epochs=tuple(
            completed_epochs
        ),
        metrics={
            "val_loss": best_val_loss,
        },
    )