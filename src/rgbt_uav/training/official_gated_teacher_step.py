"""One supervised training step for the official gated teacher."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor
from torch.optim import Optimizer

from rgbt_uav.models.official_gated_teacher import (
    OfficialGatedDualStreamTeacher,
)
from rgbt_uav.training.official_gated_teacher_loss import (
    OfficialGatedTeacherDetectionLoss,
)


class OfficialGatedTeacherStepError(RuntimeError):
    """Raised when a gated-teacher training step fails."""


def _require_image(
    batch: Mapping[str, object],
    key: str,
    *,
    channels: int,
    device: torch.device,
) -> Tensor:
    value = batch.get(key)

    if not isinstance(value, Tensor):
        raise OfficialGatedTeacherStepError(
            f"batch.{key} must be a tensor"
        )

    if (
        value.ndim != 4
        or value.shape[1] != channels
    ):
        raise OfficialGatedTeacherStepError(
            f"batch.{key} has an invalid image shape"
        )

    if not torch.isfinite(value).all():
        raise OfficialGatedTeacherStepError(
            f"batch.{key} must contain finite values"
        )

    return value.to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )


def _require_sequence(
    output: Mapping[str, object],
    key: str,
) -> tuple[Tensor, Tensor, Tensor]:
    value = output.get(key)

    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        raise OfficialGatedTeacherStepError(
            f"teacher output {key} must be a sequence"
        )

    levels = tuple(value)

    if (
        len(levels) != 3
        or not all(
            isinstance(level, Tensor)
            for level in levels
        )
    ):
        raise OfficialGatedTeacherStepError(
            f"teacher output {key} must "
            "contain three tensor levels"
        )

    return levels  # type: ignore[return-value]


def _require_loss_schema(
    losses: Mapping[str, Tensor],
) -> None:
    expected = {
        "total",
        "box",
        "classification",
        "dfl",
    }

    if set(losses) != expected:
        raise OfficialGatedTeacherStepError(
            "detection loss returned "
            "an invalid schema"
        )

    if not all(
        isinstance(value, Tensor)
        and value.ndim == 0
        and torch.isfinite(value)
        for value in losses.values()
    ):
        raise OfficialGatedTeacherStepError(
            "detection losses must be "
            "finite scalar tensors"
        )


def _gate_metrics(
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
            raise OfficialGatedTeacherStepError(
                f"{level_name} gate must have "
                "shape [B, 2, C, 1, 1]"
            )

        if not torch.isfinite(weights).all():
            raise OfficialGatedTeacherStepError(
                f"{level_name} gate must be finite"
            )

        sums = weights.sum(
            dim=1
        )

        if not torch.allclose(
            sums,
            torch.ones_like(sums),
            rtol=1e-5,
            atol=1e-6,
        ):
            raise OfficialGatedTeacherStepError(
                f"{level_name} gate weights "
                "must be normalized"
            )

        rgb_mean = float(
            weights[:, 0]
            .detach()
            .mean()
            .cpu()
        )
        ir_mean = float(
            weights[:, 1]
            .detach()
            .mean()
            .cpu()
        )

        result[
            f"gate_rgb_{level_name}"
        ] = rgb_mean
        result[
            f"gate_ir_{level_name}"
        ] = ir_mean

    return result


def run_official_gated_teacher_step(
    teacher: OfficialGatedDualStreamTeacher,
    criterion: OfficialGatedTeacherDetectionLoss,
    optimizer: Optimizer,
    batch: Mapping[str, object],
    *,
    device: torch.device,
) -> dict[str, float]:
    """Run one supervised optimizer step for the gated teacher."""

    if not isinstance(
        teacher,
        OfficialGatedDualStreamTeacher,
    ):
        raise OfficialGatedTeacherStepError(
            "teacher must be an "
            "OfficialGatedDualStreamTeacher"
        )

    if not isinstance(
        criterion,
        OfficialGatedTeacherDetectionLoss,
    ):
        raise OfficialGatedTeacherStepError(
            "criterion must be an "
            "OfficialGatedTeacherDetectionLoss"
        )

    if not isinstance(optimizer, Optimizer):
        raise OfficialGatedTeacherStepError(
            "optimizer must be a torch optimizer"
        )

    if not isinstance(batch, Mapping):
        raise OfficialGatedTeacherStepError(
            "batch must be a mapping"
        )

    if not isinstance(device, torch.device):
        raise OfficialGatedTeacherStepError(
            "device must be a torch.device"
        )

    try:
        teacher.to(device)
    except (RuntimeError, TypeError) as exc:
        raise OfficialGatedTeacherStepError(
            "could not move teacher to device"
        ) from exc

    loss_device = getattr(
        criterion.loss,
        "device",
        None,
    )

    try:
        teacher_device = next(teacher.parameters()).device
    except StopIteration as exc:
        raise OfficialGatedTeacherStepError("teacher must contain parameters") from exc

    loss_device = getattr(
        criterion.loss,
        "device",
        None,
    )

    if loss_device is None or torch.device(loss_device) != teacher_device:
        raise OfficialGatedTeacherStepError(
            "criterion and teacher must be constructed on the same device"
        )

    rgb = _require_image(
        batch,
        "rgb",
        channels=3,
        device=device,
    )
    ir = _require_image(
        batch,
        "ir",
        channels=1,
        device=device,
    )

    teacher.train()
    optimizer.zero_grad(
        set_to_none=True
    )

    try:
        output = teacher(
            rgb,
            ir,
        )
    except (
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise OfficialGatedTeacherStepError(
            "gated teacher forward failed"
        ) from exc

    if not isinstance(output, Mapping):
        raise OfficialGatedTeacherStepError(
            "gated teacher must return a mapping"
        )

    native_predictions = _require_sequence(
        output,
        "native_predictions",
    )
    gates = _require_sequence(
        output,
        "gates",
    )

    moved_batch = dict(batch)
    moved_batch["rgb"] = rgb
    moved_batch["ir"] = ir

    try:
        losses = criterion(
            native_predictions,
            moved_batch,
        )
    except (
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise OfficialGatedTeacherStepError(
            "gated teacher loss failed"
        ) from exc

    _require_loss_schema(losses)

    gate_metrics = _gate_metrics(
        gates
    )

    try:
        losses["total"].backward()
        optimizer.step()
    except RuntimeError as exc:
        raise OfficialGatedTeacherStepError(
            "gated teacher optimization failed"
        ) from exc

    metrics = {
        "total": float(
            losses["total"].detach().cpu()
        ),
        "box": float(
            losses["box"].detach().cpu()
        ),
        "classification": float(
            losses[
                "classification"
            ].detach().cpu()
        ),
        "dfl": float(
            losses["dfl"].detach().cpu()
        ),
        **gate_metrics,
    }

    if not all(
        torch.isfinite(
            torch.tensor(value)
        )
        for value in metrics.values()
    ):
        raise OfficialGatedTeacherStepError(
            "step metrics must all be finite"
        )

    return metrics