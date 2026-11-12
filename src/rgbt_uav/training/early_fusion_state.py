from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


class EarlyFusionStateError(ValueError):
    """Raised when an early-fusion training state is invalid."""


_FORMAT_VERSION = 1
_ALLOWED_STATUSES = frozenset({"running", "completed"})


@dataclass(frozen=True)
class EarlyFusionState:
    """Persisted state for a resumable early-fusion preflight run."""

    experiment_id: str
    status: str
    completed_epoch: int
    best_val_loss: float | None
    last_checkpoint: Path
    best_checkpoint: Path | None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.experiment_id, str)
            or not self.experiment_id.strip()
        ):
            raise EarlyFusionStateError(
                "experiment_id must be a non-empty string"
            )

        if self.status not in _ALLOWED_STATUSES:
            raise EarlyFusionStateError(
                "status must be either 'running' or 'completed'"
            )

        if (
            isinstance(self.completed_epoch, bool)
            or not isinstance(self.completed_epoch, int)
            or self.completed_epoch < 0
        ):
            raise EarlyFusionStateError(
                "completed_epoch must be a non-negative integer"
            )

        if self.best_val_loss is not None:
            if (
                isinstance(self.best_val_loss, bool)
                or not isinstance(self.best_val_loss, (int, float))
                or not math.isfinite(float(self.best_val_loss))
                or float(self.best_val_loss) < 0
            ):
                raise EarlyFusionStateError(
                    "best_val_loss must be a non-negative finite number or null"
                )

            object.__setattr__(
                self,
                "best_val_loss",
                float(self.best_val_loss),
            )

        last_checkpoint = Path(self.last_checkpoint)
        if not str(last_checkpoint):
            raise EarlyFusionStateError(
                "last_checkpoint must not be empty"
            )
        object.__setattr__(
            self,
            "last_checkpoint",
            last_checkpoint,
        )

        if self.best_checkpoint is not None:
            best_checkpoint = Path(self.best_checkpoint)

            if not str(best_checkpoint):
                raise EarlyFusionStateError(
                    "best_checkpoint must not be empty"
                )

            if self.best_val_loss is None:
                raise EarlyFusionStateError(
                    "best_val_loss is required when best_checkpoint is set"
                )

            object.__setattr__(
                self,
                "best_checkpoint",
                best_checkpoint,
            )

        if self.status == "completed":
            if self.completed_epoch <= 0:
                raise EarlyFusionStateError(
                    "completed state requires completed_epoch greater than zero"
                )

            if self.best_checkpoint is None:
                raise EarlyFusionStateError(
                    "completed state requires best_checkpoint"
                )

            if self.best_val_loss is None:
                raise EarlyFusionStateError(
                    "completed state requires best_val_loss"
                )

    def to_dict(self) -> dict[str, object]:
        """Return the stable JSON representation of this state."""

        return {
            "format_version": _FORMAT_VERSION,
            "experiment_id": self.experiment_id,
            "status": self.status,
            "completed_epoch": self.completed_epoch,
            "best_val_loss": self.best_val_loss,
            "last_checkpoint": str(self.last_checkpoint),
            "best_checkpoint": (
                str(self.best_checkpoint)
                if self.best_checkpoint is not None
                else None
            ),
        }

    def write_json(self, path: Path) -> None:
        """Atomically publish this state as UTF-8 JSON."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        content = (
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")

        temporary_path: Path | None = None

        try:
            temporary = tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            )
            temporary_path = Path(temporary.name)

            with temporary:
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())

            os.replace(temporary_path, destination)
            temporary_path = None
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    @classmethod
    def from_json(cls, path: Path) -> EarlyFusionState:
        """Read and validate a persisted training state."""

        source = Path(path)

        if not source.is_file():
            raise EarlyFusionStateError(
                f"state file does not exist: {source}"
            )

        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EarlyFusionStateError(
                f"could not read state file: {source}"
            ) from exc

        if not isinstance(raw, dict):
            raise EarlyFusionStateError(
                "state file root must be a JSON object"
            )

        expected_keys = {
            "format_version",
            "experiment_id",
            "status",
            "completed_epoch",
            "best_val_loss",
            "last_checkpoint",
            "best_checkpoint",
        }

        if set(raw) != expected_keys:
            missing = expected_keys - set(raw)
            unknown = set(raw) - expected_keys

            details: list[str] = []

            if missing:
                details.append(
                    "missing=" + ", ".join(sorted(missing))
                )

            if unknown:
                details.append(
                    "unknown=" + ", ".join(sorted(unknown))
                )

            raise EarlyFusionStateError(
                "state file fields do not match contract: "
                + "; ".join(details)
            )

        if raw["format_version"] != _FORMAT_VERSION:
            raise EarlyFusionStateError(
                "unsupported state format_version"
            )

        return cls(
            experiment_id=raw["experiment_id"],
            status=raw["status"],
            completed_epoch=raw["completed_epoch"],
            best_val_loss=raw["best_val_loss"],
            last_checkpoint=raw["last_checkpoint"],
            best_checkpoint=raw["best_checkpoint"],
        )