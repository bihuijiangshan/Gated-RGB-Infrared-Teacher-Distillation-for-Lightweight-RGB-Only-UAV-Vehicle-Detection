from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class EarlyFusionConfigError(ValueError):
    """Raised when an early-fusion configuration is invalid."""


_ALLOWED_KEYS = frozenset(
    {
        "experiment_id",
        "experiment_group",
        "data_root",
        "output_root",
        "model",
        "epochs",
        "batch",
        "imgsz",
        "seed",
        "device",
        "workers",
        "lr",
        "num_classes",
        "resume",
        "pretrained",
    }
)


@dataclass(frozen=True)
class EarlyFusionConfig:
    experiment_id: str
    experiment_group: str
    data_root: Path
    output_root: Path
    model: str
    epochs: int
    batch: int
    imgsz: int
    seed: int
    device: str
    workers: int
    lr: float
    num_classes: int
    resume: bool
    pretrained: Path | None

    @classmethod
    def from_yaml(cls, path: Path) -> EarlyFusionConfig:
        config_path = Path(path)

        if not config_path.is_file():
            raise EarlyFusionConfigError(
                f"configuration does not exist: {config_path}"
            )

        try:
            raw = yaml.safe_load(
                config_path.read_text(encoding="utf-8")
            )
        except (OSError, yaml.YAMLError) as exc:
            raise EarlyFusionConfigError(
                f"could not read configuration: {config_path}"
            ) from exc

        if not isinstance(raw, dict):
            raise EarlyFusionConfigError(
                "configuration root must be a mapping"
            )

        unknown = set(raw) - _ALLOWED_KEYS
        _REQUIRED_KEYS = _ALLOWED_KEYS - {"pretrained"}
        if unknown:
            names = ", ".join(sorted(str(value) for value in unknown))
            raise EarlyFusionConfigError(
                f"unknown configuration keys: {names}"
            )

        missing = _REQUIRED_KEYS - set(raw)
        if missing:
            names = ", ".join(sorted(missing))
            raise EarlyFusionConfigError(
                f"missing required configuration keys: {names}"
            )

        return cls(
            experiment_id=_string(raw, "experiment_id"),
            experiment_group=_string(raw, "experiment_group"),
            data_root=Path(_string(raw, "data_root")),
            output_root=Path(_string(raw, "output_root")),
            model=_string(raw, "model"),
            epochs=_positive_int(raw, "epochs"),
            batch=_positive_int(raw, "batch"),
            imgsz=_valid_imgsz(raw["imgsz"]),
            seed=_non_negative_int(raw, "seed"),
            device=_string(raw, "device"),
            workers=_non_negative_int(raw, "workers"),
            lr=_positive_float(raw, "lr"),
            num_classes=_positive_int(raw, "num_classes"),
            resume=_strict_bool(raw, "resume"),
            pretrained=_optional_path(raw, "pretrained"),
        )


def _string(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)

    if not isinstance(value, str) or not value.strip():
        raise EarlyFusionConfigError(
            f"{key} must be a non-empty string"
        )

    return value


def _positive_int(raw: dict[str, Any], key: str) -> int:
    value = raw.get(key)

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EarlyFusionConfigError(
            f"{key} must be a positive integer"
        )

    return value


def _non_negative_int(raw: dict[str, Any], key: str) -> int:
    value = raw.get(key)

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EarlyFusionConfigError(
            f"{key} must be a non-negative integer"
        )

    return value


def _valid_imgsz(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EarlyFusionConfigError("imgsz must be an integer")

    if value <= 0 or value % 32 != 0:
        raise EarlyFusionConfigError(
            "imgsz must be positive and divisible by 32"
        )

    return value


def _positive_float(raw: dict[str, Any], key: str) -> float:
    value = raw.get(key)

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EarlyFusionConfigError(
            f"{key} must be a positive finite number"
        )

    value = float(value)

    if not math.isfinite(value) or value <= 0:
        raise EarlyFusionConfigError(
            f"{key} must be a positive finite number"
        )

    return value


def _strict_bool(raw: dict[str, Any], key: str) -> bool:
    value = raw.get(key)

    if not isinstance(value, bool):
        raise EarlyFusionConfigError(
            f"{key} must be a boolean"
        )

    return value

def _optional_path(
    raw: dict[str, Any],
    key: str,
) -> Path | None:
    value = raw.get(key)

    if value is None:
        return None

    if not isinstance(value, str) or not value.strip():
        raise EarlyFusionConfigError(
            f"{key} must be a non-empty string or null"
        )

    return Path(value)