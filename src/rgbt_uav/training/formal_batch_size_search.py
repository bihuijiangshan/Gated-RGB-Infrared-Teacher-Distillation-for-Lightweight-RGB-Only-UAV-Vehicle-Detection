"""Bounded, train-only batch-size search for formal workloads."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from rgbt_uav.training.formal_calibration_plan import (
    FormalCalibrationPlanError,
    FormalCalibrationWorkload,
    _load_mapping,
    _portable,
    _positive_integer,
    load_formal_calibration_plan,
)
from rgbt_uav.training.formal_calibration_runner import (
    WorkloadFactory,
    _latency_summary,
    _peak_memory_bytes,
    _reset_peak_memory,
    _resolved_device,
    _sha256,
    _synchronize,
    _write_json_atomically,
)


class FormalBatchSizeSearchError(RuntimeError):
    """Raised when the bounded batch-size search cannot be completed."""


@dataclass(frozen=True)
class FormalBatchSizeSearchPlan:
    """Validated batch-size candidates and bounded execution limits."""

    protocol_id: str
    device: str
    warmup_batches: int
    measured_batches: int
    max_memory_fraction: float
    candidates: tuple[int, ...]
    output: Path
    workloads: tuple[FormalCalibrationWorkload, ...]

    def to_metadata(self) -> dict[str, object]:
        """Return a JSON-safe plan that cannot be mistaken for execution."""

        return {
            "format_version": 1,
            "status": "planned",
            "execute": False,
            "protocol_id": self.protocol_id,
            "split": "train",
            "device": self.device,
            "warmup_batches": self.warmup_batches,
            "measured_batches": self.measured_batches,
            "max_memory_fraction": self.max_memory_fraction,
            "candidates": list(self.candidates),
            "output": _portable(self.output),
            "workloads": [workload.experiment for workload in self.workloads],
        }


def load_formal_batch_size_search_plan(
    path: Path,
    matrix_path: Path,
) -> FormalBatchSizeSearchPlan:
    """Load the search plan and its already-validated workload base plan."""

    payload = _load_mapping(Path(path), label="batch-size search plan")
    if payload.get("format_version") != 1:
        raise FormalCalibrationPlanError("format_version must be 1")
    if payload.get("status") != "planned":
        raise FormalCalibrationPlanError("status must be planned")
    if payload.get("split") != "train":
        raise FormalCalibrationPlanError("split must be train")
    base_plan_value = payload.get("base_plan")
    if not isinstance(base_plan_value, str) or not base_plan_value:
        raise FormalCalibrationPlanError("base_plan must be a non-empty path")
    base_plan = load_formal_calibration_plan(
        Path(base_plan_value),
        Path(matrix_path),
    )
    if payload.get("protocol_id") != base_plan.protocol_id:
        raise FormalCalibrationPlanError(
            "protocol_id must match the calibration base plan"
        )
    device = payload.get("device")
    if not isinstance(device, str) or not device.strip():
        raise FormalCalibrationPlanError("device must be a non-empty string")
    if _resolved_device(device) != _resolved_device(base_plan.device):
        raise FormalCalibrationPlanError("device must match the base plan")
    warmup_batches = _positive_integer(payload, "warmup_batches", maximum=5)
    measured_batches = _positive_integer(
        payload,
        "measured_batches",
        maximum=10,
    )
    fraction = payload.get("max_memory_fraction")
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not 0.5 <= float(fraction) <= 0.9
    ):
        raise FormalCalibrationPlanError(
            "max_memory_fraction must be between 0.5 and 0.9"
        )
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise FormalCalibrationPlanError("candidates must be a non-empty list")
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > 256
        for value in raw_candidates
    ):
        raise FormalCalibrationPlanError(
            "candidates must contain integers between 1 and 256"
        )
    candidates = tuple(raw_candidates)
    if tuple(sorted(set(candidates))) != candidates:
        raise FormalCalibrationPlanError(
            "candidates must be unique and strictly increasing"
        )
    output_value = payload.get("output")
    if not isinstance(output_value, str) or not output_value:
        raise FormalCalibrationPlanError("output must be a non-empty path")
    output = Path(output_value)
    if not _portable(output).startswith("runs/archives/"):
        raise FormalCalibrationPlanError(
            "output must be a child of runs/archives"
        )
    return FormalBatchSizeSearchPlan(
        protocol_id=base_plan.protocol_id,
        device=device.strip(),
        warmup_batches=warmup_batches,
        measured_batches=measured_batches,
        max_memory_fraction=float(fraction),
        candidates=candidates,
        output=output,
        workloads=base_plan.workloads,
    )


def _is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.OutOfMemoryError) or (
        "out of memory" in str(exc).lower()
        and "cuda" in str(exc).lower()
    )


def _default_total_memory(device: str) -> int:
    return int(torch.cuda.get_device_properties(torch.device(device)).total_memory)


def run_formal_batch_size_search(
    plan: FormalBatchSizeSearchPlan,
    *,
    workload_factory: WorkloadFactory,
    clock: Any = time.perf_counter,
    synchronize: Any = _synchronize,
    reset_peak_memory: Any = _reset_peak_memory,
    peak_memory_bytes: Any = _peak_memory_bytes,
    total_memory_bytes: Any = _default_total_memory,
    empty_cache: Any = torch.cuda.empty_cache,
    git_metadata: dict[str, object] | None = None,
) -> dict[str, Any]:
    """Search candidates without changing formal configs or touching test data."""

    device = _resolved_device(plan.device)
    if not device.startswith("cuda"):
        raise FormalBatchSizeSearchError("batch-size search requires CUDA")
    if plan.output.exists() or plan.output.is_symlink():
        raise FormalBatchSizeSearchError(
            f"batch-size search output already exists: {plan.output}"
        )
    total_memory = total_memory_bytes(device)
    if isinstance(total_memory, bool) or not isinstance(total_memory, int):
        raise FormalBatchSizeSearchError("total CUDA memory must be an integer")
    if total_memory <= 0:
        raise FormalBatchSizeSearchError("total CUDA memory must be positive")
    memory_limit = int(total_memory * plan.max_memory_fraction)
    workload_results: list[dict[str, Any]] = []
    complete = True

    for workload in plan.workloads:
        attempts: list[dict[str, Any]] = []
        selected_batch: int | None = None
        unsafe_upper_bound: int | None = None
        for batch in plan.candidates:
            runtime = None
            durations: list[float] = []
            attempt: dict[str, Any] = {"batch": batch}
            try:
                reset_peak_memory(device)
                runtime = workload_factory(replace(workload, batch=batch), device)
                for _ in range(plan.warmup_batches):
                    runtime.step()
                synchronize(device)
                reset_peak_memory(device)
                for _ in range(plan.measured_batches):
                    synchronize(device)
                    started = clock()
                    runtime.step()
                    synchronize(device)
                    durations.append((clock() - started) * 1000.0)
                peak = peak_memory_bytes(device)
                if isinstance(peak, bool) or not isinstance(peak, int) or peak < 0:
                    raise FormalBatchSizeSearchError(
                        "peak memory must be a non-negative integer"
                    )
                attempt.update(
                    {
                        "peak_memory_bytes": peak,
                        "memory_fraction": peak / total_memory,
                        "latency": _latency_summary(durations),
                    }
                )
                if peak <= memory_limit:
                    attempt["status"] = "passed"
                    selected_batch = batch
                else:
                    attempt["status"] = "memory_limit_exceeded"
                    unsafe_upper_bound = batch
            except Exception as exc:
                if not _is_cuda_oom(exc):
                    raise FormalBatchSizeSearchError(
                        f"batch-size search failed: {workload.experiment} batch {batch}"
                    ) from exc
                attempt["status"] = "cuda_out_of_memory"
                unsafe_upper_bound = batch
            finally:
                if runtime is not None:
                    runtime.close()
                gc.collect()
                empty_cache()
            attempts.append(attempt)
            if unsafe_upper_bound is not None:
                break

        if selected_batch is None:
            raise FormalBatchSizeSearchError(
                f"no safe candidate for workload: {workload.experiment}"
            )
        if unsafe_upper_bound is None:
            complete = False
        workload_result: dict[str, Any] = {
                "experiment": workload.experiment,
                "seed": workload.seed,
                "config": _portable(workload.config),
                "config_sha256": _sha256(workload.config),
                "recommended_batch": (
                    selected_batch if unsafe_upper_bound is not None else None
                ),
                "largest_passing_candidate": selected_batch,
                "unsafe_upper_bound": unsafe_upper_bound,
                "attempts": attempts,
                "test_evaluated": False,
        }
        teacher = workload.calibration_teacher
        if teacher is not None:
            workload_result["calibration_teacher"] = {
                "kind": teacher.kind,
                "checkpoint": _portable(teacher.checkpoint),
                "sha256": teacher.sha256,
                "evidence": _portable(teacher.evidence),
            }
        workload_results.append(workload_result)

    result: dict[str, Any] = {
        "format_version": 1,
        "status": "completed" if complete else "upper_bound_not_found",
        "execute": True,
        "protocol_id": plan.protocol_id,
        "scope": "bounded train-only batch-size search",
        "split": "train",
        "device": device,
        "total_memory_bytes": total_memory,
        "max_memory_fraction": plan.max_memory_fraction,
        "memory_limit_bytes": memory_limit,
        "warmup_batches": plan.warmup_batches,
        "measured_batches": plan.measured_batches,
        "test_evaluated": False,
        "workloads": workload_results,
    }
    if git_metadata is not None:
        result["git"] = dict(git_metadata)
    _write_json_atomically(plan.output, result)
    return result
