"""Evidence orchestration for formal unimodal training."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rgbt_uav.training.formal_unimodal_backend import (
    run_ultralytics_formal_training,
)
from rgbt_uav.training.formal_unimodal_config import (
    FormalUnimodalConfig,
)
from rgbt_uav.training.formal_unimodal_plan import (
    FormalUnimodalTrainingPlan,
    build_formal_unimodal_training_plan,
)


class FormalUnimodalRunnerError(
    RuntimeError
):
    """Raised when formal unimodal training evidence is invalid."""


@dataclass(frozen=True)
class FormalUnimodalRunResult:
    """Completed formal unimodal training artifacts."""

    best_weights: Path
    last_weights: Path
    completion_path: Path
    epoch_weights: tuple[Path, ...]
    completed_epochs: tuple[int, ...]


def _portable_path(
    path: Path,
) -> str:
    return str(path).replace(
        "\\",
        "/",
    )


def _sha256(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    try:
        with path.open("rb") as handle:
            while block := handle.read(
                1024 * 1024
            ):
                digest.update(block)
    except OSError as exc:
        raise FormalUnimodalRunnerError(
            f"failed to hash artifact: {path}"
        ) from exc

    return digest.hexdigest()


def _resolve_git_commit() -> str:
    try:
        result = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (
        OSError,
        subprocess.CalledProcessError,
    ) as exc:
        raise FormalUnimodalRunnerError(
            "failed to resolve git commit"
        ) from exc

    return result.stdout.strip()


def _validate_git_commit(
    value: object,
) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(
            r"[0-9a-fA-F]{40}",
            value.strip(),
        )
        is None
    ):
        raise FormalUnimodalRunnerError(
            "git commit must be a 40-character "
            "hexadecimal digest"
        )

    return value.strip().lower()


def _write_json_atomically(
    path: Path,
    payload: dict[str, Any],
) -> None:
    temporary_path: Path | None = None

    try:
        serialized = json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        temporary = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=".",
            suffix=".completion.tmp",
            dir=path.parent,
            delete=False,
        )
        temporary_path = Path(
            temporary.name
        )
        with temporary:
            temporary.write(serialized)
            temporary.write("\n")

        os.replace(
            temporary_path,
            path,
        )
        temporary_path = None
    except (
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        raise FormalUnimodalRunnerError(
            f"failed to write completion evidence: {path}"
        ) from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(
                    missing_ok=True
                )
            except OSError:
                pass


def _require_regular_file(
    path: Path,
    *,
    label: str,
) -> None:
    if (
        not path.is_file()
        or path.is_symlink()
    ):
        raise FormalUnimodalRunnerError(
            f"{label} must be a regular non-symlink file: {path}"
        )


def _validate_input_paths(
    config: FormalUnimodalConfig,
) -> None:
    _require_regular_file(
        config.data,
        label="data",
    )

    if (
        not config.prediction_source.is_dir()
        or config.prediction_source.is_symlink()
    ):
        raise FormalUnimodalRunnerError(
            "prediction_source must be a regular "
            f"non-symlink directory: {config.prediction_source}"
        )


def _validate_epoch_history(
    plan: FormalUnimodalTrainingPlan,
    *,
    epochs: int,
) -> tuple[Path, ...]:
    epoch_root = (
        plan.training_root / "epochs"
    )

    expected = tuple(
        epoch_root
        / f"epoch-{epoch:03d}.pt"
        for epoch in range(
            1,
            epochs + 1,
        )
    )

    invalid = [
        path
        for path in expected
        if (
            not path.is_file()
            or path.is_symlink()
        )
    ]

    if invalid:
        preview = ", ".join(
            str(path)
            for path in invalid[:3]
        )
        raise FormalUnimodalRunnerError(
            "epoch checkpoint history is incomplete: "
            f"{preview}"
        )

    return expected


def _resume_checkpoint(
    config: FormalUnimodalConfig,
) -> tuple[Path, int]:
    completion_path = (
        config.output_root / "completion.json"
    )
    if completion_path.exists() or completion_path.is_symlink():
        raise FormalUnimodalRunnerError(
            "completed formal training cannot be resumed"
        )

    last_weights = (
        config.output_root
        / "train"
        / "weights"
        / "last.pt"
    )
    _require_regular_file(
        last_weights,
        label="resume last_weights",
    )

    epoch_root = (
        config.output_root
        / "train"
        / "epochs"
    )
    if not epoch_root.is_dir() or epoch_root.is_symlink():
        raise FormalUnimodalRunnerError(
            "resume epoch history must be a regular directory"
        )

    epochs: list[int] = []
    for path in epoch_root.iterdir():
        match = re.fullmatch(
            r"epoch-(\d{3})\.pt",
            path.name,
        )
        if match is None:
            continue
        _require_regular_file(
            path,
            label="resume epoch checkpoint",
        )
        epochs.append(int(match.group(1)))

    if not epochs:
        raise FormalUnimodalRunnerError(
            "resume epoch checkpoint history is empty"
        )

    completed_epoch = max(epochs)
    if sorted(epochs) != list(
        range(1, completed_epoch + 1)
    ):
        raise FormalUnimodalRunnerError(
            "resume epoch checkpoint history must be sequential"
        )
    if completed_epoch >= config.epochs:
        raise FormalUnimodalRunnerError(
            "resume target must exceed completed epoch"
        )

    latest_epoch = (
        epoch_root
        / f"epoch-{completed_epoch:03d}.pt"
    )
    if _sha256(last_weights) != _sha256(latest_epoch):
        raise FormalUnimodalRunnerError(
            "resume last_weights must match the latest epoch checkpoint"
        )

    return last_weights, completed_epoch


def run_formal_unimodal_training(
    config: FormalUnimodalConfig,
    *,
    resume: bool = False,
    training_backend: Callable[
        [FormalUnimodalTrainingPlan],
        object,
    ]
    | None = None,
    git_commit_resolver: Callable[
        [],
        str,
    ] = _resolve_git_commit,
) -> FormalUnimodalRunResult:
    """Run one backend and publish completion only after validation."""

    if not isinstance(
        config,
        FormalUnimodalConfig,
    ):
        raise FormalUnimodalRunnerError(
            "config must be a FormalUnimodalConfig"
        )

    if not isinstance(resume, bool):
        raise FormalUnimodalRunnerError(
            "resume must be boolean"
        )

    selected_backend = (
        run_ultralytics_formal_training if training_backend is None else training_backend
    )

    if not callable(selected_backend):
        raise FormalUnimodalRunnerError("training_backend must be callable")

    if not callable(git_commit_resolver):
        raise FormalUnimodalRunnerError(
            "git_commit_resolver must be callable"
        )

    _validate_input_paths(config)

    output_root = config.output_root
    resumed_from_epoch = 0
    resume_checkpoint: Path | None = None
    if resume:
        if not output_root.is_dir() or output_root.is_symlink():
            raise FormalUnimodalRunnerError(
                "resume output_root must be an existing regular directory"
            )
        (
            resume_checkpoint,
            resumed_from_epoch,
        ) = _resume_checkpoint(config)
    elif output_root.exists() or output_root.is_symlink():
        raise FormalUnimodalRunnerError(
            "output_root must not already exist"
        )

    plan = (
        build_formal_unimodal_training_plan(
            config,
            resume_checkpoint=(
                resume_checkpoint
            ),
        )
    )

    if not resume:
        try:
            output_root.mkdir(
                parents=True,
                exist_ok=False,
            )
        except OSError as exc:
            raise FormalUnimodalRunnerError(
                f"failed to create output_root: {output_root}"
            ) from exc

    try:
        selected_backend(plan)
    except Exception as exc:
        raise FormalUnimodalRunnerError(
            "formal training backend failed"
        ) from exc

    _require_regular_file(
        plan.best_weights,
        label="best_weights",
    )
    _require_regular_file(
        plan.last_weights,
        label="last_weights",
    )

    epoch_weights = (
        _validate_epoch_history(
            plan,
            epochs=config.epochs,
        )
    )
    completed_epochs = tuple(
        range(
            1,
            config.epochs + 1,
        )
    )

    git_commit = (
        _validate_git_commit(
            git_commit_resolver()
        )
    )

    completion: dict[str, Any] = {
        "format_version": 1,
        "status": "completed",
        "experiment_id": (
            config.experiment_id
        ),
        "experiment_group": (
            config.experiment_group
        ),
        "git_commit": git_commit,
        "config": config.to_metadata(),
        "selection_split": "val",
        "test_evaluated": False,
        "completed_epochs": list(
            completed_epochs
        ),
        "resumed_from_epoch": (
            resumed_from_epoch
        ),
        "training_root": (
            _portable_path(
                plan.training_root
            )
        ),
        "best_weights": (
            _portable_path(
                plan.best_weights
            )
        ),
        "last_weights": (
            _portable_path(
                plan.last_weights
            )
        ),
        "epoch_weights": [
            _portable_path(path)
            for path in epoch_weights
        ],
        "sha256": {
            "data_yaml": _sha256(
                config.data
            ),
            "best_weights": _sha256(
                plan.best_weights
            ),
            "last_weights": _sha256(
                plan.last_weights
            ),
            "epoch_weights": {
                f"{epoch:03d}": _sha256(
                    path
                )
                for epoch, path in zip(
                    completed_epochs,
                    epoch_weights,
                    strict=True,
                )
            },
        },
    }

    _write_json_atomically(
        plan.completion_path,
        completion,
    )

    return FormalUnimodalRunResult(
        best_weights=plan.best_weights,
        last_weights=plan.last_weights,
        completion_path=(
            plan.completion_path
        ),
        epoch_weights=epoch_weights,
        completed_epochs=(
            completed_epochs
        ),
    )
