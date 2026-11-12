"""Epoch orchestration for the official gated teacher."""

from __future__ import annotations

from collections.abc import (
    Callable,
    Iterable,
    Mapping,
    Sequence,
)

import torch
from torch import Tensor, nn

from rgbt_uav.training.official_gated_teacher_step import (
    run_official_gated_teacher_step,
)


class OfficialGatedTeacherEpochError(RuntimeError):
    """Raised when a gated-teacher epoch fails."""


_METRIC_KEYS = (
    "total",
    "box",
    "classification",
    "dfl",
    "gate_rgb_p3",
    "gate_ir_p3",
    "gate_rgb_p4",
    "gate_ir_p4",
    "gate_rgb_p5",
    "gate_ir_p5",
)

_LOSS_KEYS = (
    "total",
    "box",
    "classification",
    "dfl",
)


def _batch_size(
    batch: Mapping[str, object],
) -> int:
    rgb = batch.get("rgb")

    if not isinstance(rgb, Tensor):
        raise OfficialGatedTeacherEpochError(
            "batch.rgb must be a tensor"
        )

    if (
        rgb.ndim != 4
        or rgb.shape[0] <= 0
    ):
        raise OfficialGatedTeacherEpochError(
            "batch.rgb must have a positive "
            "BCHW batch dimension"
        )

    return int(rgb.shape[0])


def _finite_float(
    value: object,
    *,
    label: str,
) -> float:
    if isinstance(value, Tensor):
        if value.ndim != 0:
            raise OfficialGatedTeacherEpochError(
                f"{label} must be scalar"
            )

        result = float(
            value.detach().cpu()
        )
    elif (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
    ):
        result = float(value)
    else:
        raise OfficialGatedTeacherEpochError(
            f"{label} must be numeric"
        )

    if not torch.isfinite(
        torch.tensor(result)
    ):
        raise OfficialGatedTeacherEpochError(
            f"{label} must be finite"
        )

    return result


def _require_three_tensors(
    value: object,
    *,
    label: str,
) -> tuple[Tensor, Tensor, Tensor]:
    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        raise OfficialGatedTeacherEpochError(
            f"{label} must be a sequence"
        )

    levels = tuple(value)

    if (
        len(levels) != 3
        or not all(
            isinstance(level, Tensor)
            for level in levels
        )
    ):
        raise OfficialGatedTeacherEpochError(
            f"{label} must contain "
            "three tensor levels"
        )

    return levels  # type: ignore[return-value]


def _validation_gate_metrics(
    gates: tuple[Tensor, Tensor, Tensor],
) -> dict[str, float]:
    result: dict[str, float] = {}

    for level_name, weights in zip(
        ("p3", "p4", "p5"),
        gates,
        strict=True,
    ):
        if (
            weights.ndim != 5
            or weights.shape[1] != 2
        ):
            raise OfficialGatedTeacherEpochError(
                f"{level_name} gate has "
                "an invalid shape"
            )

        if not torch.isfinite(
            weights
        ).all():
            raise OfficialGatedTeacherEpochError(
                f"{level_name} gate must be finite"
            )

        gate_sum = weights.sum(
            dim=1
        )

        if not torch.allclose(
            gate_sum,
            torch.ones_like(gate_sum),
            rtol=1e-5,
            atol=1e-6,
        ):
            raise OfficialGatedTeacherEpochError(
                f"{level_name} gate weights "
                "must be normalized"
            )

        result[
            f"gate_rgb_{level_name}"
        ] = float(
            weights[:, 0]
            .detach()
            .mean()
            .cpu()
        )
        result[
            f"gate_ir_{level_name}"
        ] = float(
            weights[:, 1]
            .detach()
            .mean()
            .cpu()
        )

    return result


def _weighted_result(
    totals: Mapping[str, float],
    *,
    samples: int,
    batches: int,
) -> dict[str, float]:
    if batches <= 0 or samples <= 0:
        raise OfficialGatedTeacherEpochError(
            "loader produced no batches"
        )

    return {
        key: totals[key] / samples
        for key in _METRIC_KEYS
    } | {
        "batches": float(batches),
        "samples": float(samples),
    }


def train_official_gated_teacher_epoch(
    teacher: object,
    criterion: object,
    optimizer: object,
    loader: Iterable[
        Mapping[str, object]
    ],
    *,
    device: torch.device,
    step_fn: Callable[
        ...,
        Mapping[str, float],
    ] = run_official_gated_teacher_step,
) -> dict[str, float]:
    """Train the official gated teacher for one epoch."""

    if not isinstance(
        device,
        torch.device,
    ):
        raise OfficialGatedTeacherEpochError(
            "device must be a torch.device"
        )

    totals = {
        key: 0.0
        for key in _METRIC_KEYS
    }
    batches = 0
    samples = 0

    for batch in loader:
        if not isinstance(batch, Mapping):
            raise OfficialGatedTeacherEpochError(
                f"training batch {batches} "
                "must be a mapping"
            )

        size = _batch_size(batch)

        try:
            metrics = step_fn(
                teacher,
                criterion,
                optimizer,
                batch,
                device=device,
            )
        except (
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            raise OfficialGatedTeacherEpochError(
                "training batch failed at "
                f"batch {batches}"
            ) from exc

        if set(metrics) != set(
            _METRIC_KEYS
        ):
            raise OfficialGatedTeacherEpochError(
                "training step returned "
                "an invalid metric schema"
            )

        for key in _METRIC_KEYS:
            value = _finite_float(
                metrics[key],
                label=(
                    f"training metric {key}"
                ),
            )
            totals[key] += value * size

        samples += size
        batches += 1

    return _weighted_result(
        totals,
        samples=samples,
        batches=batches,
    )


def validate_official_gated_teacher_epoch(
    teacher: nn.Module,
    criterion: object,
    loader: Iterable[
        Mapping[str, object]
    ],
    *,
    device: torch.device,
) -> dict[str, float]:
    """Validate official gated-teacher loss without gradients."""

    if not isinstance(teacher, nn.Module):
        raise OfficialGatedTeacherEpochError(
            "teacher must be a torch module"
        )

    if not isinstance(
        device,
        torch.device,
    ):
        raise OfficialGatedTeacherEpochError(
            "device must be a torch.device"
        )

    try:
        teacher.to(device)
    except (RuntimeError, TypeError) as exc:
        raise OfficialGatedTeacherEpochError(
            "could not move teacher to device"
        ) from exc

    teacher.eval()

    totals = {
        key: 0.0
        for key in _METRIC_KEYS
    }
    batches = 0
    samples = 0

    with torch.no_grad():
        for batch in loader:
            if not isinstance(
                batch,
                Mapping,
            ):
                raise OfficialGatedTeacherEpochError(
                    f"validation batch {batches} "
                    "must be a mapping"
                )

            size = _batch_size(batch)
            rgb = batch.get("rgb")
            ir = batch.get("ir")

            if (
                not isinstance(rgb, Tensor)
                or rgb.ndim != 4
                or rgb.shape[1] != 3
            ):
                raise OfficialGatedTeacherEpochError(
                    "validation RGB input "
                    "must have shape [B, 3, H, W]"
                )

            if (
                not isinstance(ir, Tensor)
                or ir.ndim != 4
                or ir.shape[1] != 1
            ):
                raise OfficialGatedTeacherEpochError(
                    "validation IR input "
                    "must have shape [B, 1, H, W]"
                )

            moved_rgb = rgb.to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            moved_ir = ir.to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            moved_batch = dict(batch)
            moved_batch["rgb"] = moved_rgb
            moved_batch["ir"] = moved_ir

            try:
                output = teacher(
                    moved_rgb,
                    moved_ir,
                )
            except (
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                raise OfficialGatedTeacherEpochError(
                    "validation forward failed "
                    f"at batch {batches}"
                ) from exc

            if not isinstance(
                output,
                Mapping,
            ):
                raise OfficialGatedTeacherEpochError(
                    "teacher output must "
                    "be a mapping"
                )

            predictions = (
                _require_three_tensors(
                    output.get(
                        "native_predictions"
                    ),
                    label=(
                        "native predictions"
                    ),
                )
            )
            gates = _require_three_tensors(
                output.get("gates"),
                label="gates",
            )

            try:
                losses = criterion(
                    predictions,
                    moved_batch,
                )
            except (
                KeyError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                raise OfficialGatedTeacherEpochError(
                    "validation loss failed "
                    f"at batch {batches}"
                ) from exc

            if set(losses) != set(
                _LOSS_KEYS
            ):
                raise OfficialGatedTeacherEpochError(
                    "validation loss returned "
                    "an invalid metric schema"
                )

            metrics = {
                key: _finite_float(
                    losses[key],
                    label=(
                        f"validation metric {key}"
                    ),
                )
                for key in _LOSS_KEYS
            }
            metrics.update(
                _validation_gate_metrics(
                    gates
                )
            )

            for key in _METRIC_KEYS:
                totals[key] += (
                    metrics[key] * size
                )

            samples += size
            batches += 1

    return _weighted_result(
        totals,
        samples=samples,
        batches=batches,
    )