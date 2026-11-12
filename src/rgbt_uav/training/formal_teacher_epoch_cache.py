"""Integrity-bound reuse of one teacher epoch evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from pathlib import Path


class TeacherEpochCacheError(RuntimeError):
    """Raised when an epoch result cannot be safely created or reused."""


METRICS = ("mAP50-95", "mAP50", "AP75", "AP_small", "AR_small")


def require(condition, message):
    if not condition:
        raise TeacherEpochCacheError(message)


def file_sha256(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"Unsafe file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value):
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise TeacherEpochCacheError(f"Invalid JSON: {path}") from exc


def validate_report(report, checkpoint, epoch):
    require(isinstance(report, dict), "Report must be a mapping")
    require(
        type(report.get("restored_epoch")) is int
        and report["restored_epoch"] == epoch,
        "Report epoch mismatch",
    )
    checkpoint_value = report.get("checkpoint")
    require(isinstance(checkpoint_value, str), "Report checkpoint is missing")
    require(
        Path(checkpoint_value).resolve() == checkpoint.resolve(),
        "Report checkpoint mismatch",
    )
    for name in METRICS:
        value = report.get(name)
        require(
            type(value) in (int, float)
            and math.isfinite(value)
            and 0 <= value <= 1,
            f"Invalid metric: {name}",
        )


def evaluate_bound_epoch(
    *,
    checkpoint,
    output,
    epoch,
    context,
    evaluation_fn,
    evaluation_kwargs,
):
    """Return (report, reused); context must bind data, code and settings."""
    checkpoint = Path(checkpoint).absolute()
    output = Path(output).absolute()
    receipt = output.with_suffix(output.suffix + ".receipt.json")
    require(type(epoch) is int and epoch > 0, "Epoch must be positive")
    require(isinstance(context, dict) and context, "Context must be non-empty")
    require(not output.is_symlink(), "Output must not be a symlink")
    require(not receipt.is_symlink(), "Receipt must not be a symlink")
    require(
        checkpoint.resolve() not in (output.resolve(), receipt.resolve()),
        "Output overlaps checkpoint",
    )

    # Copy context through JSON so later caller mutations cannot alter it.
    frozen_context = json.loads(json_bytes(context))
    binding = {
        "format_version": 1,
        "epoch": epoch,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "context": frozen_context,
    }

    if output.exists() or receipt.exists():
        require(
            output.is_file() and receipt.is_file(),
            "Incomplete cache; preserve files and inspect",
        )
        saved = read_json(receipt)
        require(isinstance(saved, dict), "Receipt must be a mapping")
        require(saved.get("binding") == binding, "Cache binding mismatch")
        require(
            saved.get("metrics_sha256") == file_sha256(output),
            "Metrics hash mismatch",
        )
        report = read_json(output)
        validate_report(report, checkpoint, epoch)
        require(
            file_sha256(checkpoint) == binding["checkpoint_sha256"],
            "Checkpoint changed during cache verification",
        )
        return report, True

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".teacher-epoch-", dir=output.parent
    ) as temporary:
        staged = Path(temporary) / output.name
        kwargs = dict(evaluation_kwargs)
        kwargs.update(checkpoint=checkpoint, output=staged)
        evaluation_fn(**kwargs)
        staged_hash = file_sha256(staged)
        raw = staged.read_bytes()
        require(
            hashlib.sha256(raw).hexdigest() == staged_hash,
            "Staged metrics changed",
        )
        report = read_json(staged)
        require(json.loads(raw) == report, "Staged metrics changed")
        validate_report(report, checkpoint, epoch)
        require(
            file_sha256(checkpoint) == binding["checkpoint_sha256"],
            "Checkpoint changed during evaluation",
        )
        saved = {"binding": binding, "metrics_sha256": staged_hash}
        require(
            not output.exists()
            and not output.is_symlink()
            and not receipt.exists()
            and not receipt.is_symlink(),
            "Output appeared during evaluation",
        )

        # Exclusive creation: never replace an existing result.
        # A crash between these writes leaves an incomplete cache,
        # which the next invocation rejects instead of silently reusing.
        with output.open("xb") as stream:
            stream.write(raw)
        with receipt.open("xb") as stream:
            stream.write(json_bytes(saved))

    require(file_sha256(output) == staged_hash, "Published metrics mismatch")
    require(read_json(receipt) == saved, "Published receipt mismatch")
    return report, False