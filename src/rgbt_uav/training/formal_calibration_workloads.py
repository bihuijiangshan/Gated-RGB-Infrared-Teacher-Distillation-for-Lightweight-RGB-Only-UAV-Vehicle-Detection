"""Workload dispatch and loader adapters for formal calibration."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from rgbt_uav.training.formal_calibration_plan import (
    FormalCalibrationWorkload,
)
from rgbt_uav.training.formal_calibration_runner import (
    FormalCalibrationRuntime,
    WorkloadFactory,
)


class FormalCalibrationWorkloadError(RuntimeError):
    """Raised when a calibration workload adapter is invalid."""


FORMAL_CALIBRATION_EXPERIMENTS = (
    "rgb_only",
    "ir_only",
    "early_fusion_teacher",
    "gated_teacher",
    "lightweight_detection_only",
    "lightweight_early_fusion_distillation",
    "lightweight_gated_distillation",
)


WorkloadAdapter = Callable[[FormalCalibrationWorkload, str], FormalCalibrationRuntime]


def build_loader_calibration_runtime(
    loader: Iterable[Any],
    *,
    train_batch: Callable[[Any], object],
    close: Callable[[], None],
) -> FormalCalibrationRuntime:
    """Adapt a finite loader into a repeatable one-training-batch runtime."""

    if not callable(train_batch):
        raise FormalCalibrationWorkloadError("train_batch must be callable")
    if not callable(close):
        raise FormalCalibrationWorkloadError("close must be callable")

    iterator = iter(loader)

    def step() -> object:
        nonlocal iterator
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            try:
                batch = next(iterator)
            except StopIteration as exc:
                raise FormalCalibrationWorkloadError(
                    "calibration loader must not be empty"
                ) from exc
        return train_batch(batch)

    return FormalCalibrationRuntime(step=step, close=close)


def build_formal_workload_factory(
    adapters: Mapping[str, WorkloadAdapter],
) -> WorkloadFactory:
    """Build strict dispatch for the seven representative workloads."""

    expected = set(FORMAL_CALIBRATION_EXPERIMENTS)
    supplied = set(adapters)
    missing = expected - supplied
    unexpected = supplied - expected
    if missing:
        raise FormalCalibrationWorkloadError(
            "missing workload adapters: " + ", ".join(sorted(missing))
        )
    if unexpected:
        raise FormalCalibrationWorkloadError(
            "unexpected workload adapters: " + ", ".join(sorted(unexpected))
        )
    if any(not callable(adapter) for adapter in adapters.values()):
        raise FormalCalibrationWorkloadError(
            "every workload adapter must be callable"
        )

    frozen_adapters = dict(adapters)

    def factory(
        workload: FormalCalibrationWorkload,
        device: str,
    ) -> FormalCalibrationRuntime:
        adapter = frozen_adapters.get(workload.experiment)
        if adapter is None:
            raise FormalCalibrationWorkloadError(
                f"unsupported calibration experiment: {workload.experiment}"
            )
        runtime = adapter(workload, device)
        if not isinstance(runtime, FormalCalibrationRuntime):
            raise FormalCalibrationWorkloadError(
                f"adapter returned an invalid runtime: {workload.experiment}"
            )
        return runtime

    return factory
