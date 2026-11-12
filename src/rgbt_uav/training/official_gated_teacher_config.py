"""Configuration for official gated-teacher training."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


class OfficialGatedTeacherConfigError(
    ValueError
):
    """Raised when a gated-teacher config is invalid."""


@dataclass(frozen=True)
class OfficialGatedTeacherConfig:
    """Validated gated-teacher experiment configuration."""

    experiment_id: str
    experiment_group: str
    data_root: Path
    output_root: Path
    model: str
    pretrained: Path
    epochs: int
    batch: int
    imgsz: int
    seed: int
    device: str
    workers: int
    lr: float
    weight_decay: float
    num_classes: int
    resume: bool

    def __post_init__(self) -> None:
        for name in (
            "experiment_id",
            "experiment_group",
            "model",
            "device",
        ):
            value = getattr(self, name)

            if (
                not isinstance(value, str)
                or not value.strip()
            ):
                raise (
                    OfficialGatedTeacherConfigError(
                        f"{name} must be a "
                        "non-empty string"
                    )
                )

        for name in (
            "data_root",
            "output_root",
            "pretrained",
        ):
            if not isinstance(
                getattr(self, name),
                Path,
            ):
                raise (
                    OfficialGatedTeacherConfigError(
                        f"{name} must be a path"
                    )
                )

        for name in (
            "epochs",
            "batch",
            "num_classes",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise (
                    OfficialGatedTeacherConfigError(
                        f"{name} must be a "
                        "positive integer"
                    )
                )

        if (
            isinstance(self.imgsz, bool)
            or not isinstance(
                self.imgsz,
                int,
            )
            or self.imgsz <= 0
            or self.imgsz % 32 != 0
        ):
            raise OfficialGatedTeacherConfigError(
                "imgsz must be a positive "
                "integer divisible by 32"
            )

        for name in (
            "seed",
            "workers",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise (
                    OfficialGatedTeacherConfigError(
                        f"{name} must be a "
                        "non-negative integer"
                    )
                )

        if (
            isinstance(self.lr, bool)
            or not isinstance(
                self.lr,
                (int, float),
            )
            or not math.isfinite(
                float(self.lr)
            )
            or float(self.lr) <= 0
        ):
            raise OfficialGatedTeacherConfigError(
                "lr must be a positive "
                "finite number"
            )

        if (
            isinstance(
                self.weight_decay,
                bool,
            )
            or not isinstance(
                self.weight_decay,
                (int, float),
            )
            or not math.isfinite(
                float(self.weight_decay)
            )
            or float(self.weight_decay) < 0
        ):
            raise OfficialGatedTeacherConfigError(
                "weight_decay must be a "
                "non-negative finite number"
            )

        if not isinstance(
            self.resume,
            bool,
        ):
            raise OfficialGatedTeacherConfigError(
                "resume must be boolean"
            )

    @classmethod
    def from_yaml(
        cls,
        path: Path,
    ) -> OfficialGatedTeacherConfig:
        """Load and strictly validate a YAML configuration."""

        source = Path(path)

        if not source.is_file():
            raise OfficialGatedTeacherConfigError(
                "config must be a regular "
                f"file: {source}"
            )

        try:
            payload: Any = yaml.safe_load(
                source.read_text(
                    encoding="utf-8"
                )
            )
        except (
            OSError,
            UnicodeError,
            yaml.YAMLError,
        ) as exc:
            raise OfficialGatedTeacherConfigError(
                "failed to read config: "
                f"{source}"
            ) from exc

        if not isinstance(payload, dict):
            raise OfficialGatedTeacherConfigError(
                "config root must be a mapping"
            )

        allowed = {
            field.name
            for field in fields(cls)
        }
        supplied = set(payload)
        unknown = supplied - allowed
        missing = allowed - supplied

        if unknown:
            names = ", ".join(
                sorted(
                    str(name)
                    for name in unknown
                )
            )
            raise OfficialGatedTeacherConfigError(
                "config contains unknown "
                f"fields: {names}"
            )

        if missing:
            names = ", ".join(
                sorted(missing)
            )
            raise OfficialGatedTeacherConfigError(
                "config is missing fields: "
                f"{names}"
            )

        converted = dict(payload)

        for name in (
            "data_root",
            "output_root",
            "pretrained",
        ):
            value = converted[name]

            if (
                not isinstance(value, str)
                or not value.strip()
            ):
                raise OfficialGatedTeacherConfigError(
                    f"{name} must be a "
                    "non-empty path"
                )

            converted[name] = Path(
                value
            )

        try:
            return cls(**converted)
        except TypeError as exc:
            raise OfficialGatedTeacherConfigError(
                "config contains invalid "
                "value types"
            ) from exc

    def to_metadata(
        self,
    ) -> dict[str, object]:
        """Return reproducibility metadata for evidence files."""

        return {
            "model": self.model,
            "pretrained": str(
                self.pretrained
            ),
            "epochs": self.epochs,
            "batch": self.batch,
            "imgsz": self.imgsz,
            "seed": self.seed,
            "device": self.device,
            "workers": self.workers,
            "lr": float(self.lr),
            "weight_decay": float(
                self.weight_decay
            ),
            "num_classes": (
                self.num_classes
            ),
            "resume": self.resume,
        }