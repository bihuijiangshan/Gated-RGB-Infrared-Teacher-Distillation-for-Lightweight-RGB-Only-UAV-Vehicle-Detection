from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class DistillationConfigError(ValueError):
    """Raised when a distillation configuration is invalid."""


_ALLOWED_KEYS = {
    "experiment_id",
    "experiment_group",
    "data_root",
    "output_root",
    "epochs",
    "batch",
    "imgsz",
    "seed",
    "device",
    "workers",
    "lr",
    "num_classes",
    "resume",
    "teacher_base_channels",
    "student_base_channels",
    "reg_max",
    "feature_weight",
    "logit_weight",
    "temperature",
    "teacher_checkpoint",
    "teacher_kind",
    "teacher_model",
}


@dataclass(frozen=True)
class DistillationConfig:
    experiment_id: str
    experiment_group: str
    data_root: Path
    output_root: Path
    epochs: int
    batch: int
    imgsz: int
    seed: int
    device: str
    workers: int
    lr: float
    num_classes: int
    resume: bool
    teacher_base_channels: int
    student_base_channels: int
    reg_max: int
    feature_weight: float
    logit_weight: float
    temperature: float
    teacher_checkpoint: Path | None
    teacher_kind: str = "custom"
    teacher_model: str = "yolov8n.yaml"

    @classmethod
    def from_yaml(cls, path: Path) -> DistillationConfig:
        source = Path(path)

        if not source.is_file():
            raise DistillationConfigError(
                f"configuration does not exist: {source}"
            )

        try:
            raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            raise DistillationConfigError(
                f"could not read configuration: {source}"
            ) from exc

        if not isinstance(raw, dict):
            raise DistillationConfigError(
                "configuration root must be a mapping"
            )

        unknown = set(raw) - _ALLOWED_KEYS
        optional_keys = {
            "teacher_kind",
            "teacher_model",
        }
        required_keys = _ALLOWED_KEYS - optional_keys
        missing = required_keys - set(raw)

        if unknown:
            raise DistillationConfigError(
                "unknown configuration keys: "
                + ", ".join(sorted(str(value) for value in unknown))
            )

        if missing:
            raise DistillationConfigError(
                "missing required configuration keys: "
                + ", ".join(sorted(missing))
            )

        teacher_checkpoint = _optional_path(
            raw,
            "teacher_checkpoint",
        )

        if teacher_checkpoint is None:
            raise DistillationConfigError(
                "teacher_checkpoint is required for distillation"
            )

        teacher_kind = _optional_string(
            raw,
            "teacher_kind",
            default="custom",
        )

        if teacher_kind not in {
            "custom",
            "official_early_fusion",
        }:
            raise DistillationConfigError(
                "teacher_kind must be 'custom' "
                "or 'official_early_fusion'"
            )

        teacher_model = _optional_string(
            raw,
            "teacher_model",
            default="yolov8n.yaml",
        )

        return cls(
            experiment_id=_string(raw, "experiment_id"),
            experiment_group=_string(raw, "experiment_group"),
            data_root=Path(_string(raw, "data_root")),
            output_root=Path(_string(raw, "output_root")),
            epochs=_positive_int(raw, "epochs"),
            batch=_positive_int(raw, "batch"),
            imgsz=_imgsz(raw["imgsz"]),
            seed=_non_negative_int(raw, "seed"),
            device=_string(raw, "device"),
            workers=_non_negative_int(raw, "workers"),
            lr=_positive_float(raw, "lr"),
            num_classes=_positive_int(raw, "num_classes"),
            resume=_bool(raw, "resume"),
            teacher_base_channels=_positive_int(
                raw,
                "teacher_base_channels",
            ),
            student_base_channels=_positive_int(
                raw,
                "student_base_channels",
            ),
            reg_max=_positive_int(raw, "reg_max"),
            feature_weight=_non_negative_float(raw, "feature_weight"),
            logit_weight=_non_negative_float(raw, "logit_weight"),
            temperature=_positive_float(raw, "temperature"),
            teacher_checkpoint=teacher_checkpoint,
            teacher_kind=teacher_kind,
            teacher_model=teacher_model,
        )


def _string(raw: dict[str, Any], key: str) -> str:
    value = raw[key]
    if not isinstance(value, str) or not value.strip():
        raise DistillationConfigError(
            f"{key} must be a non-empty string"
        )
    return value


def _positive_int(raw: dict[str, Any], key: str) -> int:
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DistillationConfigError(
            f"{key} must be a positive integer"
        )
    return value


def _non_negative_int(raw: dict[str, Any], key: str) -> int:
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DistillationConfigError(
            f"{key} must be a non-negative integer"
        )
    return value


def _imgsz(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value % 32 != 0
    ):
        raise DistillationConfigError(
            "imgsz must be positive and divisible by 32"
        )
    return value


def _bool(raw: dict[str, Any], key: str) -> bool:
    value = raw[key]
    if not isinstance(value, bool):
        raise DistillationConfigError(
            f"{key} must be a boolean"
        )
    return value


def _positive_float(raw: dict[str, Any], key: str) -> float:
    value = raw[key]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise DistillationConfigError(
            f"{key} must be a positive finite number"
        )
    return float(value)


def _non_negative_float(raw: dict[str, Any], key: str) -> float:
    value = raw[key]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise DistillationConfigError(
            f"{key} must be a non-negative finite number"
        )
    return float(value)


def _optional_path(raw: dict[str, Any], key: str) -> Path | None:
    value = raw[key]

    if value is None:
        return None

    if not isinstance(value, str) or not value.strip():
        raise DistillationConfigError(
            f"{key} must be a non-empty string or null"
        )

    return Path(value)

def _optional_string(
    raw: dict[str, Any],
    key: str,
    *,
    default: str,
) -> str:
    value = raw.get(key, default)

    if not isinstance(value, str) or not value.strip():
        raise DistillationConfigError(
            f"{key} must be a non-empty string"
        )

    return value