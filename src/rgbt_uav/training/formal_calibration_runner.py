"""Bounded execution core for formal training calibration."""

from __future__ import annotations

import json
import math
import os
import statistics
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch

from rgbt_uav.training.formal_calibration_plan import (
    FormalCalibrationPlan,
    FormalCalibrationWorkload,
)


class FormalCalibrationRunnerError(RuntimeError):
    """Raised when bounded calibration cannot be completed safely."""


@dataclass(frozen=True)
class FormalCalibrationRuntime:
    """One initialized workload whose ``step`` runs one training batch."""

    step: Callable[[], object]
    close: Callable[[], None]


WorkloadFactory = Callable[
    [FormalCalibrationWorkload, str],
    FormalCalibrationRuntime,
]


def _validate_execution_plan(plan: FormalCalibrationPlan) -> None:
    if plan.split != "train":
        raise FormalCalibrationRunnerError("calibration split must be train")
    limits = (
        ("warmup_batches", plan.warmup_batches, 10),
        ("measured_batches", plan.measured_batches, 50),
        ("repeats", plan.repeats, 3),
    )
    for label, value, maximum in limits:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            or value > maximum
        ):
            raise FormalCalibrationRunnerError(
                f"{label} must be between 1 and {maximum}"
            )
    if not plan.workloads:
        raise FormalCalibrationRunnerError("workloads must not be empty")
    for workload in plan.workloads:
        if not workload.config.is_file() or workload.config.is_symlink():
            raise FormalCalibrationRunnerError(
                f"workload config must be a regular file: {workload.config}"
            )


def _portable(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_device(device: str) -> str:
    value = device.strip().lower()
    if value.isdigit():
        return f"cuda:{value}"
    return value


def _synchronize(device: str) -> None:
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise FormalCalibrationRunnerError("CUDA is not available")
        torch.cuda.synchronize(torch.device(device))


def _reset_peak_memory(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(device))


def _peak_memory_bytes(device: str) -> int:
    if device.startswith("cuda"):
        return int(torch.cuda.max_memory_allocated(torch.device(device)))
    return 0


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _latency_summary(values: list[float]) -> dict[str, float]:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise FormalCalibrationRunnerError(
            "measured batch durations must be finite and positive"
        )
    mean = statistics.fmean(values)
    return {
        "mean_ms": mean,
        "sample_std_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median_ms": statistics.median(values),
        "p95_ms": _percentile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
        "batches_per_second": 1000.0 / mean,
    }


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except (OSError, TypeError, ValueError) as exc:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise FormalCalibrationRunnerError(
            f"failed to write calibration evidence: {path}"
        ) from exc


def run_formal_calibration(
    plan: FormalCalibrationPlan,
    *,
    workload_factory: WorkloadFactory,
    clock: Callable[[], float] = time.perf_counter,
    synchronize: Callable[[str], None] = _synchronize,
    reset_peak_memory: Callable[[str], None] = _reset_peak_memory,
    peak_memory_bytes: Callable[[str], int] = _peak_memory_bytes,
    git_metadata: dict[str, object] | None = None,
) -> dict[str, Any]:
    """Execute only the bounded training batches declared by ``plan``."""

    _validate_execution_plan(plan)
    output = Path(plan.output)
    if output.exists() or output.is_symlink():
        raise FormalCalibrationRunnerError(
            f"calibration output already exists: {output}"
        )
    device = _resolved_device(plan.device)
    workload_results: list[dict[str, Any]] = []

    for workload in plan.workloads:
        try:
            runtime = workload_factory(workload, device)
        except Exception as exc:
            raise FormalCalibrationRunnerError(
                f"failed to initialize workload: {workload.experiment}"
            ) from exc
        if not isinstance(runtime, FormalCalibrationRuntime):
            raise FormalCalibrationRunnerError(
                "workload_factory must return FormalCalibrationRuntime"
            )

        durations: list[float] = []
        repeat_peaks: list[int] = []
        try:
            for _ in range(plan.warmup_batches):
                runtime.step()
            synchronize(device)

            for _ in range(plan.repeats):
                reset_peak_memory(device)
                for _ in range(plan.measured_batches):
                    synchronize(device)
                    started = clock()
                    runtime.step()
                    synchronize(device)
                    duration_ms = (clock() - started) * 1000.0
                    durations.append(duration_ms)
                peak = peak_memory_bytes(device)
                if isinstance(peak, bool) or not isinstance(peak, int) or peak < 0:
                    raise FormalCalibrationRunnerError(
                        "peak memory must be a non-negative integer"
                    )
                repeat_peaks.append(peak)
        except FormalCalibrationRunnerError:
            raise
        except Exception as exc:
            raise FormalCalibrationRunnerError(
                f"workload execution failed: {workload.experiment}"
            ) from exc
        finally:
            runtime.close()

        workload_result: dict[str, Any] = {
                "experiment": workload.experiment,
                "seed": workload.seed,
                "config": _portable(workload.config),
                "config_sha256": _sha256(workload.config),
                "warmup_batches": plan.warmup_batches,
                "measured_batches_per_repeat": plan.measured_batches,
                "repeats": plan.repeats,
                "total_measured_batches": len(durations),
                "latency": _latency_summary(durations),
                "repeat_peak_memory_bytes": repeat_peaks,
                "peak_memory_bytes": max(repeat_peaks),
                "test_evaluated": False,
        }
        if workload.batch is not None:
            workload_result["batch"] = workload.batch
        calibration_teacher = workload.calibration_teacher
        if calibration_teacher is not None:
            workload_result["calibration_teacher"] = {
                "kind": calibration_teacher.kind,
                "checkpoint": _portable(calibration_teacher.checkpoint),
                "sha256": calibration_teacher.sha256,
                "evidence": _portable(calibration_teacher.evidence),
            }
        workload_results.append(workload_result)

    result: dict[str, Any] = {
        "format_version": 1,
        "status": "completed",
        "execute": True,
        "protocol_id": plan.protocol_id,
        "scope": "bounded training-step runtime calibration",
        "split": plan.split,
        "device": device,
        "test_evaluated": False,
        "workloads": workload_results,
    }
    if git_metadata is not None:
        result["git"] = dict(git_metadata)
    _write_json_atomically(output, result)
    return result
