"""Strict configuration for formal RGB-only and IR-only runs."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


class FormalUnimodalConfigError(
    ValueError
):
    """Raised when a formal unimodal config is invalid."""


def _normalized_path(
    value: Path | str,
) -> str:
    return str(value).replace(
        "\\",
        "/",
    ).rstrip("/")


@dataclass(frozen=True)
class FormalUnimodalConfig:
    """Validated formal RGB-only or IR-only configuration."""

    experiment_id: str
    experiment_group: str
    modality: str
    model: str
    data: Path
    prediction_source: Path
    output_root: Path
    epochs: int
    imgsz: int
    rect: bool
    batch: int
    seed: int
    workers: int
    device: str | int
    amp: bool
    cache: bool
    optimizer: str
    lr: float
    weight_decay: float

    def __post_init__(self) -> None:
        for name in (
            "experiment_id",
            "experiment_group",
            "model",
            "optimizer",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not value.strip()
            ):
                raise FormalUnimodalConfigError(
                    f"{name} must be a non-empty string"
                )

        if self.modality not in {
            "rgb",
            "ir",
        }:
            raise FormalUnimodalConfigError(
                "modality must be 'rgb' or 'ir'"
            )

        for name in (
            "data",
            "prediction_source",
            "output_root",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, Path)
                or not str(value)
            ):
                raise FormalUnimodalConfigError(
                    f"{name} must be a non-empty path"
                )

        if (
            isinstance(self.epochs, bool)
            or not isinstance(
                self.epochs,
                int,
            )
            or self.epochs != 100
        ):
            raise FormalUnimodalConfigError(
                "epochs must be exactly 100"
            )

        for name in (
            "imgsz",
            "batch",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise FormalUnimodalConfigError(
                    f"{name} must be a positive integer"
                )

        if self.imgsz % 32 != 0:
            raise FormalUnimodalConfigError(
                "imgsz must be divisible by 32"
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
                raise FormalUnimodalConfigError(
                    f"{name} must be a non-negative integer"
                )

        if not (
            isinstance(self.device, str)
            and self.device.strip()
            or isinstance(self.device, int)
            and not isinstance(
                self.device,
                bool,
            )
            and self.device >= 0
        ):
            raise FormalUnimodalConfigError(
                "device must be a non-empty string "
                "or non-negative integer"
            )

        for name in (
            "rect",
            "amp",
            "cache",
        ):
            if not isinstance(
                getattr(self, name),
                bool,
            ):
                raise FormalUnimodalConfigError(
                    f"{name} must be boolean"
                )

        for name in (
            "lr",
            "weight_decay",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(
                    value,
                    (int, float),
                )
                or not math.isfinite(
                    float(value)
                )
            ):
                raise FormalUnimodalConfigError(
                    f"{name} must be finite"
                )

        if float(self.lr) <= 0:
            raise FormalUnimodalConfigError(
                "lr must be positive"
            )

        if float(self.weight_decay) < 0:
            raise FormalUnimodalConfigError(
                "weight_decay must be non-negative"
            )

        expected_data = (
            "data/processed/"
            "yolo_full_irref_with_test/"
            f"{self.modality}.yaml"
        )
        if (
            _normalized_path(self.data)
            != expected_data
        ):
            raise FormalUnimodalConfigError(
                "data must match modality and use "
                "the full frozen dataset"
            )

        expected_prediction_source = (
            "data/processed/"
            "yolo_full_irref_with_test/"
            f"{self.modality}/images/val"
        )
        if (
            _normalized_path(
                self.prediction_source
            )
            != expected_prediction_source
        ):
            raise FormalUnimodalConfigError(
                "prediction_source must match modality "
                "and use the validation split"
            )

        output_root = _normalized_path(
            self.output_root
        )
        if (
            not output_root.startswith(
                "runs/formal/"
            )
            or output_root == "runs/formal"
        ):
            raise FormalUnimodalConfigError(
                "output_root must be a strict child "
                "of runs/formal"
            )

    @classmethod
    def from_yaml(
        cls,
        path: Path,
    ) -> FormalUnimodalConfig:
        """Load and strictly validate one YAML config."""

        source = Path(path)

        if not source.is_file():
            raise FormalUnimodalConfigError(
                f"config must be a regular file: {source}"
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
            raise FormalUnimodalConfigError(
                f"failed to read config: {source}"
            ) from exc

        if not isinstance(payload, dict):
            raise FormalUnimodalConfigError(
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
            raise FormalUnimodalConfigError(
                "config contains unknown fields: "
                f"{names}"
            )

        if missing:
            names = ", ".join(
                sorted(missing)
            )
            raise FormalUnimodalConfigError(
                "config is missing fields: "
                f"{names}"
            )

        converted = dict(payload)

        for name in (
            "data",
            "prediction_source",
            "output_root",
        ):
            value = converted[name]
            if (
                not isinstance(value, str)
                or not value.strip()
            ):
                raise FormalUnimodalConfigError(
                    f"{name} must be a non-empty path"
                )
            converted[name] = Path(value)

        try:
            return cls(**converted)
        except TypeError as exc:
            raise FormalUnimodalConfigError(
                "config contains invalid value types"
            ) from exc

    def to_metadata(
        self,
    ) -> dict[str, object]:
        """Return immutable training metadata."""

        return {
            "modality": self.modality,
            "model": self.model,
            "data": _normalized_path(
                self.data
            ),
            "prediction_source": (
                _normalized_path(
                    self.prediction_source
                )
            ),
            "epochs": self.epochs,
            "imgsz": self.imgsz,
            "rect": self.rect,
            "batch": self.batch,
            "seed": self.seed,
            "workers": self.workers,
            "device": self.device,
            "amp": self.amp,
            "cache": self.cache,
            "optimizer": self.optimizer,
            "lr": float(self.lr),
            "weight_decay": float(
                self.weight_decay
            ),
        }