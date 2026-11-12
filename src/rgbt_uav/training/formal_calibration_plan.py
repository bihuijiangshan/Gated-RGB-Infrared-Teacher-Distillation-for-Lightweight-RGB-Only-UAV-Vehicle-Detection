"""Safe bounded workload plan for formal training calibration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml


class FormalCalibrationPlanError(ValueError):
    """Raised when a formal calibration plan violates its safety contract."""


@dataclass(frozen=True)
class FormalCalibrationTeacher:
    """A calibration-only teacher artifact with pinned provenance."""

    kind: str
    checkpoint: Path
    sha256: str
    evidence: Path


@dataclass(frozen=True)
class FormalCalibrationWorkload:
    """One representative formal experiment workload."""

    experiment: str
    seed: int
    config: Path
    calibration_teacher: FormalCalibrationTeacher | None = None
    batch: int | None = None


@dataclass(frozen=True)
class FormalCalibrationPlan:
    """Validated bounded calibration plan."""

    protocol_id: str
    split: str
    device: str
    warmup_batches: int
    measured_batches: int
    repeats: int
    output: Path
    workloads: tuple[FormalCalibrationWorkload, ...]

    def to_metadata(self) -> dict[str, object]:
        """Return a JSON-serializable plan without executing workloads."""

        return {
            "status": "planned",
            "execute": False,
            "protocol_id": self.protocol_id,
            "split": self.split,
            "device": self.device,
            "warmup_batches": self.warmup_batches,
            "measured_batches": self.measured_batches,
            "repeats": self.repeats,
            "output": _portable(self.output),
            "workloads": [
                _workload_metadata(workload)
                for workload in self.workloads
            ],
        }


def _workload_metadata(
    workload: FormalCalibrationWorkload,
) -> dict[str, object]:
    result: dict[str, object] = {
                    "experiment": workload.experiment,
                    "seed": workload.seed,
                    "config": _portable(workload.config),
    }
    teacher = workload.calibration_teacher
    if teacher is not None:
        result["calibration_teacher"] = {
            "kind": teacher.kind,
            "checkpoint": _portable(teacher.checkpoint),
            "sha256": teacher.sha256,
            "evidence": _portable(teacher.evidence),
        }
    if workload.batch is not None:
        result["batch"] = workload.batch
    return result


def _portable(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def _load_mapping(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FormalCalibrationPlanError(
            f"{label} must be a regular non-symlink file: {path}"
        )
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise FormalCalibrationPlanError(f"failed to load {label}") from exc
    if not isinstance(payload, dict):
        raise FormalCalibrationPlanError(f"{label} root must be a mapping")
    return payload


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_CALIBRATION_TEACHER_KINDS = {
    "lightweight_detection_only": "official_early_fusion",
    "lightweight_early_fusion_distillation": "official_early_fusion",
    "lightweight_gated_distillation": "official_gated",
}


def _load_calibration_teacher(
    raw: Mapping[str, Any],
    *,
    experiment: str,
    index: int,
) -> FormalCalibrationTeacher | None:
    value = raw.get("calibration_teacher")
    expected_kind = _CALIBRATION_TEACHER_KINDS.get(experiment)
    if expected_kind is None:
        if value is not None:
            raise FormalCalibrationPlanError(
                f"workload {index} must not define calibration_teacher"
            )
        return None
    if not isinstance(value, Mapping):
        raise FormalCalibrationPlanError(
            f"workload {index} calibration_teacher must be a mapping"
        )
    kind = value.get("kind")
    checkpoint_value = value.get("checkpoint")
    expected_sha256 = value.get("sha256")
    evidence_value = value.get("evidence")
    if kind != expected_kind:
        raise FormalCalibrationPlanError(
            f"workload {index} calibration teacher kind is invalid"
        )
    if not isinstance(checkpoint_value, str) or not checkpoint_value:
        raise FormalCalibrationPlanError(
            f"workload {index} calibration teacher checkpoint is invalid"
        )
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or expected_sha256 != expected_sha256.lower()
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise FormalCalibrationPlanError(
            f"workload {index} calibration teacher sha256 is invalid"
        )
    if not isinstance(evidence_value, str) or not evidence_value:
        raise FormalCalibrationPlanError(
            f"workload {index} calibration teacher evidence is invalid"
        )
    checkpoint = Path(checkpoint_value)
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise FormalCalibrationPlanError(
            f"workload {index} calibration teacher checkpoint "
            f"must be a regular non-symlink file: {checkpoint}"
        )
    if _file_sha256(checkpoint) != expected_sha256:
        raise FormalCalibrationPlanError(
            f"workload {index} calibration teacher sha256 does not match"
        )
    evidence = Path(evidence_value)
    evidence_payload = _load_mapping(
        evidence,
        label=f"workload {index} calibration teacher evidence",
    )
    if evidence_payload.get("status") != "completed":
        raise FormalCalibrationPlanError(
            f"workload {index} calibration teacher evidence is not completed"
        )
    evidence_checkpoint = evidence_payload.get(
        "selected_checkpoint",
        evidence_payload.get("best_weights"),
    )
    if _portable(evidence_checkpoint or "") != _portable(checkpoint):
        raise FormalCalibrationPlanError(
            f"workload {index} calibration teacher evidence checkpoint mismatch"
        )
    evidence_sha256 = evidence_payload.get("sha256")
    if evidence_sha256 is not None and evidence_sha256 != expected_sha256:
        raise FormalCalibrationPlanError(
            f"workload {index} calibration teacher evidence sha256 mismatch"
        )
    return FormalCalibrationTeacher(
        kind=kind,
        checkpoint=checkpoint,
        sha256=expected_sha256,
        evidence=evidence,
    )


def _positive_integer(
    payload: Mapping[str, Any],
    field: str,
    *,
    maximum: int,
) -> int:
    value = payload.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > maximum
    ):
        raise FormalCalibrationPlanError(
            f"{field} must be between 1 and {maximum}"
        )
    return value


def load_formal_calibration_plan(
    plan_path: Path,
    matrix_path: Path,
) -> FormalCalibrationPlan:
    """Load a bounded plan and verify every workload against the matrix."""

    payload = _load_mapping(Path(plan_path), label="calibration plan")
    matrix = _load_mapping(Path(matrix_path), label="formal matrix")

    if payload.get("format_version") != 1:
        raise FormalCalibrationPlanError("format_version must be 1")
    if payload.get("status") != "planned":
        raise FormalCalibrationPlanError("status must be planned")
    if payload.get("protocol_id") != matrix.get("protocol_id"):
        raise FormalCalibrationPlanError("protocol_id must match matrix")
    if payload.get("split") != "train":
        raise FormalCalibrationPlanError("split must be train")

    device = payload.get("device")
    if not isinstance(device, str) or not device.strip():
        raise FormalCalibrationPlanError("device must be a non-empty string")

    warmup_batches = _positive_integer(
        payload,
        "warmup_batches",
        maximum=10,
    )
    measured_batches = _positive_integer(
        payload,
        "measured_batches",
        maximum=50,
    )
    repeats = _positive_integer(payload, "repeats", maximum=3)

    output_value = payload.get("output")
    if not isinstance(output_value, str) or not output_value.strip():
        raise FormalCalibrationPlanError("output must be a non-empty path")
    output = Path(output_value)
    normalized_output = _portable(output).rstrip("/")
    if not normalized_output.startswith("runs/archives/"):
        raise FormalCalibrationPlanError(
            "output must be a child of runs/archives"
        )

    runs = matrix.get("runs")
    if not isinstance(runs, list):
        raise FormalCalibrationPlanError("matrix runs must be a list")
    declared = {
        (
            run.get("experiment"),
            run.get("seed"),
            _portable(run.get("config", "")),
        )
        for run in runs
        if isinstance(run, Mapping)
    }

    raw_workloads = payload.get("workloads")
    if not isinstance(raw_workloads, list) or not raw_workloads:
        raise FormalCalibrationPlanError("workloads must be a non-empty list")

    workloads: list[FormalCalibrationWorkload] = []
    experiments: set[str] = set()
    for index, raw in enumerate(raw_workloads):
        if not isinstance(raw, Mapping):
            raise FormalCalibrationPlanError(
                f"workload {index} must be a mapping"
            )
        experiment = raw.get("experiment")
        seed = raw.get("seed")
        config_value = raw.get("config")
        batch_value = raw.get("batch")
        if not isinstance(experiment, str) or not experiment:
            raise FormalCalibrationPlanError(
                f"workload {index} experiment is invalid"
            )
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise FormalCalibrationPlanError(f"workload {index} seed is invalid")
        if not isinstance(config_value, str) or not config_value:
            raise FormalCalibrationPlanError(
                f"workload {index} config is invalid"
            )
        if batch_value is not None and (
            isinstance(batch_value, bool)
            or not isinstance(batch_value, int)
            or batch_value <= 0
            or batch_value > 256
        ):
            raise FormalCalibrationPlanError(
                f"workload {index} batch must be between 1 and 256"
            )
        if experiment in experiments:
            raise FormalCalibrationPlanError("workload experiments must be unique")
        experiments.add(experiment)

        normalized_config = _portable(config_value)
        if (experiment, seed, normalized_config) not in declared:
            raise FormalCalibrationPlanError(
                f"workload {index} is not declared in the formal matrix"
            )
        config = Path(config_value)
        _load_mapping(config, label=f"workload {index} config")
        calibration_teacher = _load_calibration_teacher(
            raw,
            experiment=experiment,
            index=index,
        )
        workloads.append(
            FormalCalibrationWorkload(
                experiment=experiment,
                seed=seed,
                config=config,
                calibration_teacher=calibration_teacher,
                batch=batch_value,
            )
        )

    serialized = repr(payload).replace("\\", "/").lower()
    if "/images/test" in serialized or "/labels/test" in serialized:
        raise FormalCalibrationPlanError("calibration must not reference test data")

    return FormalCalibrationPlan(
        protocol_id=str(payload["protocol_id"]),
        split="train",
        device=device.strip(),
        warmup_batches=warmup_batches,
        measured_batches=measured_batches,
        repeats=repeats,
        output=output,
        workloads=tuple(workloads),
    )
