"""Select the best official RGB checkpoint by evaluated COCO mAP."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Iterable
from pathlib import Path


class OfficialRGBCheckpointSelectionError(RuntimeError):
    """Raised when evaluated epoch checkpoints cannot be selected."""


_REQUIRED_METRICS = (
    "mAP50-95",
    "mAP50",
    "AP75",
    "AP_small",
    "AR_small",
)


def _read_metrics(
    path: Path,
) -> dict[str, object]:
    if not path.is_file():
        raise OfficialRGBCheckpointSelectionError(
            f"metrics must be a regular file: {path}"
        )

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as exc:
        raise OfficialRGBCheckpointSelectionError(
            f"failed to read metrics: {path}"
        ) from exc

    if not isinstance(payload, dict):
        raise OfficialRGBCheckpointSelectionError(
            f"metrics root must be a mapping: {path}"
        )

    return payload


def _finite_float(
    payload: dict[str, object],
    name: str,
) -> float:
    value = payload.get(name)

    if not isinstance(value, (int, float)):
        raise OfficialRGBCheckpointSelectionError(
            f"metric {name} must be numeric"
        )

    result = float(value)
    if not math.isfinite(result):
        raise OfficialRGBCheckpointSelectionError(
            f"metric {name} must be finite"
        )

    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def select_best_map_checkpoint(
    metrics_paths: Iterable[Path],
    *,
    destination: Path,
    evidence_path: Path,
) -> dict[str, object]:
    """Copy the highest-mAP epoch checkpoint to a stable destination."""

    candidates: list[dict[str, object]] = []
    seen_epochs: set[int] = set()

    for metrics_path_value in metrics_paths:
        metrics_path = Path(metrics_path_value)
        payload = _read_metrics(metrics_path)

        epoch = payload.get("restored_epoch")
        if not isinstance(epoch, int) or epoch <= 0:
            raise OfficialRGBCheckpointSelectionError(
                "restored_epoch must be positive"
            )
        if epoch in seen_epochs:
            raise OfficialRGBCheckpointSelectionError(
                f"duplicate evaluated epoch: {epoch}"
            )
        seen_epochs.add(epoch)

        checkpoint_value = payload.get("checkpoint")
        if (
            not isinstance(checkpoint_value, str)
            or not checkpoint_value
        ):
            raise OfficialRGBCheckpointSelectionError(
                "checkpoint path is missing from metrics"
            )

        checkpoint = Path(checkpoint_value)
        if not checkpoint.is_file():
            raise OfficialRGBCheckpointSelectionError(
                f"checkpoint must be a regular file: {checkpoint}"
            )

        metrics = {
            name: _finite_float(payload, name)
            for name in _REQUIRED_METRICS
        }

        candidates.append(
            {
                "epoch": epoch,
                "checkpoint": checkpoint,
                "metrics_path": metrics_path,
                **metrics,
            }
        )

    if not candidates:
        raise OfficialRGBCheckpointSelectionError(
            "no evaluated epoch metrics were provided"
        )

    selected = max(
        candidates,
        key=lambda candidate: (
            float(candidate["mAP50-95"]),
            float(candidate["AP75"]),
            float(candidate["AP_small"]),
            -int(candidate["epoch"]),
        ),
    )

    source = Path(selected["checkpoint"])
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)

    if source.resolve() != target.resolve():
        try:
            shutil.copy2(source, target)
        except OSError as exc:
            raise OfficialRGBCheckpointSelectionError(
                f"failed to copy selected checkpoint: {target}"
            ) from exc

    digest = _sha256(target)

    evidence = {
        "format_version": 1,
        "status": "completed",
        "selection_metric": "mAP50-95",
        "selected_epoch": int(selected["epoch"]),
        "source_checkpoint": str(source),
        "selected_checkpoint": str(target),
        "sha256": digest,
        "mAP50-95": float(selected["mAP50-95"]),
        "mAP50": float(selected["mAP50"]),
        "AP75": float(selected["AP75"]),
        "AP_small": float(selected["AP_small"]),
        "AR_small": float(selected["AR_small"]),
        "evaluated_epochs": sorted(seen_epochs),
        "candidates": [
            {
                "epoch": int(candidate["epoch"]),
                "checkpoint": str(
                    candidate["checkpoint"]
                ),
                "metrics_path": str(
                    candidate["metrics_path"]
                ),
                **{
                    name: float(candidate[name])
                    for name in _REQUIRED_METRICS
                },
            }
            for candidate in sorted(
                candidates,
                key=lambda candidate: int(
                    candidate["epoch"]
                ),
            )
        ],
    }

    evidence_destination = Path(evidence_path)
    evidence_destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = evidence_destination.with_suffix(
        evidence_destination.suffix + ".tmp"
    )

    try:
        temporary.write_text(
            json.dumps(
                evidence,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(evidence_destination)
    except (OSError, TypeError, ValueError) as exc:
        if temporary.is_file():
            temporary.unlink()
        raise OfficialRGBCheckpointSelectionError(
            "failed to write checkpoint selection evidence"
        ) from exc

    return {
        "checkpoint": target,
        "selected_epoch": int(selected["epoch"]),
        "mAP50-95": float(selected["mAP50-95"]),
        "mAP50": float(selected["mAP50"]),
        "evidence_path": evidence_destination,
    }