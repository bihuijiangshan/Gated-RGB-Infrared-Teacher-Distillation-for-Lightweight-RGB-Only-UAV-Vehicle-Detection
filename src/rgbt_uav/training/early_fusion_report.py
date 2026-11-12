from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from rgbt_uav.training.early_fusion_evidence import (
    validate_early_fusion_completion,
)


class EarlyFusionReportError(ValueError):
    """Raised when an early-fusion report cannot be built."""


def _file_metadata(path: Path, field: str) -> dict[str, object]:
    try:
        if path.is_symlink() or not path.is_file():
            raise EarlyFusionReportError(
                f"{field} must be a regular file: {path}"
            )

        digest = hashlib.sha256()
        size = 0

        with path.open("rb") as handle:
            for chunk in iter(
                lambda: handle.read(1024 * 1024),
                b"",
            ):
                digest.update(chunk)
                size += len(chunk)

    except OSError as exc:
        raise EarlyFusionReportError(
            f"could not read {field}: {path}"
        ) from exc

    return {
        "path": str(path),
        "bytes": size,
        "sha256": digest.hexdigest(),
    }


def _resolve_artifact_path(
    declared: object,
    *,
    completion_path: Path,
) -> Path:
    if not isinstance(declared, str) or not declared.strip():
        raise EarlyFusionReportError(
            "artifact path must be a non-empty string"
        )

    declared_path = Path(declared)

    if declared_path.is_absolute():
        return declared_path

    candidates = (
        Path.cwd() / declared_path,
        completion_path.parent / declared_path,
    )

    for candidate in candidates:
        if candidate.is_file() and not candidate.is_symlink():
            return candidate

    return candidates[0]


def build_early_fusion_report(
    completion_path: Path,
) -> dict[str, object]:
    """Build a reproducible summary with artifact hashes."""

    source = Path(completion_path)
    payload = validate_early_fusion_completion(source)

    best_path = _resolve_artifact_path(
        payload["best_weights"],
        completion_path=source,
    )
    last_path = _resolve_artifact_path(
        payload["last_weights"],
        completion_path=source,
    )

    best_metadata = _file_metadata(
        best_path,
        "best_weights",
    )
    last_metadata = _file_metadata(
        last_path,
        "last_weights",
    )

    try:
        completion_bytes = source.read_bytes()
    except OSError as exc:
        raise EarlyFusionReportError(
            f"could not read completion: {source}"
        ) from exc

    completion_digest = hashlib.sha256(
        completion_bytes
    ).hexdigest()

    return {
        "status": payload["status"],
        "experiment_id": payload["experiment_id"],
        "experiment_group": payload["experiment_group"],
        "config": payload["config"],
        "dataset": payload["dataset"],
        "completed_epochs": payload["completed_epochs"],
        "all_completed_epochs": payload["all_completed_epochs"],
        "metrics": payload["metrics"],
        "artifacts": {
            "completion_json_sha256": completion_digest,
            "best_weights": best_metadata,
            "last_weights": last_metadata,
            "best_weights_sha256": best_metadata["sha256"],
            "last_weights_sha256": last_metadata["sha256"],
            "best_weights_bytes": best_metadata["bytes"],
            "last_weights_bytes": last_metadata["bytes"],
        },
    }


def write_early_fusion_report(
    completion_path: Path,
    destination: Path,
) -> Path:
    """Build and atomically write a JSON report."""

    report = build_early_fusion_report(completion_path)
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None

    try:
        temporary = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        )
        temporary_path = Path(temporary.name)

        with temporary:
            json.dump(
                report,
                temporary,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
                indent=2,
            )
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())

        os.replace(temporary_path, output)
        temporary_path = None
    except OSError as exc:
        raise EarlyFusionReportError(
            f"could not write report: {output}"
        ) from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass

    return output