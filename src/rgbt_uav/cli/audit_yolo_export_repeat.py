"""Audit two independently generated paired YOLO exports."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rgbt_uav.data.yolo_repeat_audit import (
    YoloRepeatAuditError,
    audit_yolo_export_repeat,
)


class YoloRepeatAuditCliError(ValueError):
    """Raised when audit evidence cannot be published safely."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolved(path: Path) -> Path:
    try:
        return Path(path).resolve(strict=False)
    except OSError as exc:
        raise YoloRepeatAuditCliError(
            f"could not resolve path: {path}"
        ) from exc


def _is_within(
    path: Path,
    root: Path,
) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_output(
    output: Path,
    *,
    first: Path,
    repeat: Path,
) -> Path:
    resolved = _resolved(output)
    first_root = _resolved(first)
    repeat_root = _resolved(repeat)

    if (
        _is_within(resolved, first_root)
        or _is_within(resolved, repeat_root)
    ):
        raise YoloRepeatAuditCliError(
            "audit output must be outside "
            "both export roots"
        )

    if resolved.exists():
        raise YoloRepeatAuditCliError(
            f"audit output already exists: {resolved}"
        )

    parent = resolved.parent
    parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        not parent.is_dir()
        or parent.is_symlink()
    ):
        raise YoloRepeatAuditCliError(
            f"audit output parent must be "
            f"a regular directory: {parent}"
        )

    return resolved


def _publish_json(
    output: Path,
    payload: dict[str, Any],
) -> None:
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())

        try:
            os.link(
                temporary_path,
                output,
            )
        except FileExistsError as exc:
            raise YoloRepeatAuditCliError(
                f"audit output already exists: {output}"
            ) from exc
        except OSError as exc:
            raise YoloRepeatAuditCliError(
                f"could not publish audit output: {output}"
            ) from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(
                    missing_ok=True
                )
            except OSError:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two independently generated "
            "paired YOLO exports"
        )
    )
    parser.add_argument(
        "--first",
        type=Path,
        required=True,
        help="First YOLO export root",
    )
    parser.add_argument(
        "--repeat",
        type=Path,
        required=True,
        help="Repeated YOLO export root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New audit JSON destination",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
) -> None:
    args = build_parser().parse_args(argv)

    output = _validate_output(
        args.output,
        first=args.first,
        repeat=args.repeat,
    )

    try:
        audit = audit_yolo_export_repeat(
            args.first,
            args.repeat,
        )
    except YoloRepeatAuditError:
        raise

    payload: dict[str, Any] = {
        "format_version": 1,
        "verified_at_utc": _utc_now(),
        **audit,
    }

    _publish_json(
        output,
        payload,
    )

    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()