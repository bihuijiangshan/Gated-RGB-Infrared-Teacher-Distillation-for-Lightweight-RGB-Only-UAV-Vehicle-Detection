"""Configuration for official RGB-T teacher to RGB student distillation."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


class OfficialRGBDistillationConfigError(ValueError):
    """Raised when an official RGB distillation config is invalid."""


@dataclass(frozen=True)
class OfficialRGBDistillationConfig:
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
    teacher_model: str
    teacher_checkpoint: Path
    student_model: str
    student_pretrained: Path
    feature_weight: float
    logit_weight: float
    temperature: float
    teacher_kind: str = (
        "official_early_fusion"
    )
    student_pretrained_kind: str = "exact"
    output_distillation_kind: str = "legacy_joint"
    classification_distillation_weight: (
        float | None
    ) = None
    dfl_distillation_weight: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "experiment_id",
            "experiment_group",
            "device",
            "teacher_model",
            "student_model",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise OfficialRGBDistillationConfigError(
                    f"{name} must be a non-empty string"
                )

            if self.output_distillation_kind not in {
                "legacy_joint",
                "detection_aware",
            }:
                raise OfficialRGBDistillationConfigError(
                    "output_distillation_kind must be "
                    "'legacy_joint' or 'detection_aware'"
                )

            for name in (
                "classification_distillation_weight",
                "dfl_distillation_weight",
            ):
                value = getattr(self, name)
                if value is not None and (
                    not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0
                ):
                    raise OfficialRGBDistillationConfigError(
                        f"{name} must be a non-negative number or null"
                    )

        for name in (
            "data_root",
            "output_root",
            "teacher_checkpoint",
            "student_pretrained",
        ):
            value = getattr(self, name)
            if not isinstance(value, Path):
                raise OfficialRGBDistillationConfigError(
                    f"{name} must be a path"
                )

        for name in (
            "epochs",
            "batch",
            "imgsz",
            "num_classes",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise OfficialRGBDistillationConfigError(
                    f"{name} must be positive"
                )

        if self.student_pretrained_kind not in {
            "exact",
            "width_reduced",
        }:
            raise OfficialRGBDistillationConfigError(
                "student_pretrained_kind must be "
                "'exact' or 'width_reduced'"
            )

        if not isinstance(self.seed, int) or self.seed < 0:
            raise OfficialRGBDistillationConfigError(
                "seed must be non-negative"
            )
        if not isinstance(self.workers, int) or self.workers < 0:
            raise OfficialRGBDistillationConfigError(
                "workers must be non-negative"
            )
        if not isinstance(self.lr, (int, float)) or self.lr <= 0:
            raise OfficialRGBDistillationConfigError(
                "lr must be positive"
            )

        for name in (
            "feature_weight",
            "logit_weight",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or value < 0
            ):
                raise OfficialRGBDistillationConfigError(
                    f"{name} must be non-negative"
                )

        if (
            not isinstance(self.temperature, (int, float))
            or self.temperature <= 0
        ):
            raise OfficialRGBDistillationConfigError(
                "temperature must be positive"
            )
        if not isinstance(self.resume, bool):
            raise OfficialRGBDistillationConfigError(
                "resume must be boolean"
            )

        if self.teacher_kind not in {
            "official_early_fusion",
            "official_gated",
        }:
            raise (
                OfficialRGBDistillationConfigError(
                    "teacher_kind must be "
                    "'official_early_fusion' "
                    "or 'official_gated'"
                )
            )

    @property
    def effective_classification_distillation_weight(
        self,
    ) -> float:
        value = self.classification_distillation_weight
        if value is None:
            return float(self.logit_weight)
        return float(value)

    @property
    def effective_dfl_distillation_weight(
        self,
    ) -> float:
        value = self.dfl_distillation_weight
        if value is None:
            return float(self.logit_weight)
        return float(value)

    
    @classmethod
    def from_yaml(
        cls,
        path: Path,
    ) -> OfficialRGBDistillationConfig:
        source = Path(path)
        if not source.is_file():
            raise OfficialRGBDistillationConfigError(
                f"config must be a regular file: {source}"
            )

        try:
            payload: Any = yaml.safe_load(
                source.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            raise OfficialRGBDistillationConfigError(
                f"failed to read config: {source}"
            ) from exc

        if not isinstance(payload, dict):
            raise OfficialRGBDistillationConfigError(
                "config root must be a mapping"
            )

        allowed = {
            field.name
            for field in fields(cls)
        }
        unknown = set(payload) - allowed
        optional = {
            "teacher_kind",
            "student_pretrained_kind",
            "output_distillation_kind",
            "classification_distillation_weight",
            "dfl_distillation_weight",
        }
        missing = (
                allowed
                - optional
                - set(payload)
        )

        if unknown:
            names = ", ".join(sorted(unknown))
            raise OfficialRGBDistillationConfigError(
                f"config contains unknown fields: {names}"
            )
        if missing:
            names = ", ".join(sorted(missing))
            raise OfficialRGBDistillationConfigError(
                f"config is missing fields: {names}"
            )

        converted = dict(payload)
        converted.setdefault(
            "teacher_kind",
            "official_early_fusion",
        )
        converted.setdefault(
            "student_pretrained_kind",
            "exact",
        )
        converted.setdefault(
            "output_distillation_kind",
            "legacy_joint",
        )
        converted.setdefault(
            "classification_distillation_weight",
            None,
        )
        converted.setdefault(
            "dfl_distillation_weight",
            None,
        )
        for name in (
            "data_root",
            "output_root",
            "teacher_checkpoint",
            "student_pretrained",
        ):
            value = converted[name]
            if not isinstance(value, str) or not value.strip():
                raise OfficialRGBDistillationConfigError(
                    f"{name} must be a non-empty path"
                )
            converted[name] = Path(value)

        try:
            return cls(**converted)
        except TypeError as exc:
            raise OfficialRGBDistillationConfigError(
                "config contains invalid value types"
            ) from exc

    def to_metadata(self) -> dict[str, object]:
        return {
            "epochs": self.epochs,
            "batch": self.batch,
            "imgsz": self.imgsz,
            "seed": self.seed,
            "device": self.device,
            "workers": self.workers,
            "lr": float(self.lr),
            "num_classes": self.num_classes,
            "resume": self.resume,
            "teacher_kind": self.teacher_kind,
            "teacher_model": self.teacher_model,
            "teacher_checkpoint": str(self.teacher_checkpoint),
            "student_model": self.student_model,
            "student_pretrained": str(self.student_pretrained),
            "student_pretrained_kind": (self.student_pretrained_kind),
            "feature_weight": float(self.feature_weight),
            "logit_weight": float(self.logit_weight),
            "temperature": float(self.temperature),
            "output_distillation_kind": (self.output_distillation_kind),
            "classification_distillation_weight": (
                self.effective_classification_distillation_weight
            ),
            "dfl_distillation_weight": (self.effective_dfl_distillation_weight),
        }