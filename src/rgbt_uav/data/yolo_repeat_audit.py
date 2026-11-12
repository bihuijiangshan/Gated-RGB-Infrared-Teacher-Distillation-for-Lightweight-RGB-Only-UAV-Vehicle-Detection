"""Audit two independently generated paired YOLO exports."""

from __future__ import annotations

import hashlib
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

_YAML_FILES = {
    "rgb.yaml",
    "ir.yaml",
}
_MODALITIES = (
    "rgb",
    "ir",
)
_SPLITS = (
    "train",
    "val",
    "test",
)
_CLASSES = [
    "car",
    "truck",
    "bus",
    "van",
    "freight_car",
]
_CHUNK_SIZE = 1024 * 1024


class YoloRepeatAuditError(ValueError):
    """Raised when repeated YOLO exports are not equivalent."""


def _is_reparse_point(path: Path) -> bool:
    try:
        attributes = getattr(
            path.lstat(),
            "st_file_attributes",
            0,
        )
    except OSError as exc:
        raise YoloRepeatAuditError(
            f"could not inspect path: {path}"
        ) from exc

    reparse_attribute = getattr(
        stat,
        "FILE_ATTRIBUTE_REPARSE_POINT",
        0,
    )
    return bool(
        reparse_attribute
        and attributes & reparse_attribute
    )


def _validated_root(
    path: Path,
    *,
    label: str,
) -> Path:
    root = Path(path)

    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise YoloRepeatAuditError(
            f"{label} export root does not exist: {root}"
        ) from exc

    if (
        not resolved.is_dir()
        or resolved.is_symlink()
        or _is_reparse_point(resolved)
    ):
        raise YoloRepeatAuditError(
            f"{label} export root must be "
            f"a regular directory: {resolved}"
        )

    return resolved


def _file_map(
    root: Path,
    *,
    label: str,
) -> dict[str, Path]:
    files: dict[str, Path] = {}

    try:
        entries = root.rglob("*")
        for path in entries:
            relative = path.relative_to(root).as_posix()

            if (
                path.is_symlink()
                or _is_reparse_point(path)
            ):
                raise YoloRepeatAuditError(
                    f"{label} export contains "
                    f"a reparse point: {relative}"
                )

            if path.is_dir():
                continue

            if not path.is_file():
                raise YoloRepeatAuditError(
                    f"{label} export contains "
                    f"a non-regular entry: {relative}"
                )

            files[relative] = path
    except YoloRepeatAuditError:
        raise
    except OSError as exc:
        raise YoloRepeatAuditError(
            f"could not enumerate {label} export: {root}"
        ) from exc

    if not files:
        raise YoloRepeatAuditError(
            f"{label} export contains no files"
        )

    return files


def _load_yaml(
    root: Path,
    *,
    modality: str,
) -> dict[str, Any]:
    yaml_path = root / f"{modality}.yaml"

    if not yaml_path.is_file():
        raise YoloRepeatAuditError(
            f"missing {modality}.yaml"
        )

    try:
        payload = yaml.safe_load(
            yaml_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise YoloRepeatAuditError(
            f"could not load {modality}.yaml"
        ) from exc

    if not isinstance(payload, Mapping):
        raise YoloRepeatAuditError(
            f"{modality}.yaml must contain a mapping"
        )

    result = dict(payload)

    declared_root = result.get("path")
    if (
        not isinstance(declared_root, str)
        or not declared_root
    ):
        raise YoloRepeatAuditError(
            f"{modality}.yaml path must be a string"
        )

    try:
        resolved_declared_root = Path(
            declared_root
        ).resolve(strict=False)
    except OSError as exc:
        raise YoloRepeatAuditError(
            f"could not resolve {modality}.yaml path"
        ) from exc

    if resolved_declared_root != (
        root
        / modality
    ).resolve(strict=False):
        raise YoloRepeatAuditError(
            f"{modality}.yaml path does not "
            "match its export root"
        )

    for split in _SPLITS:
        if result.get(split) != f"images/{split}":
            if split == "test":
                raise YoloRepeatAuditError(
                    f"{modality}.yaml does not declare "
                    "the test split"
                )
            raise YoloRepeatAuditError(
                f"{modality}.yaml has invalid "
                f"{split} split"
            )

    if result.get("nc") != len(_CLASSES):
        raise YoloRepeatAuditError(
            f"{modality}.yaml has invalid nc"
        )

    if result.get("names") != _CLASSES:
        raise YoloRepeatAuditError(
            f"{modality}.yaml has invalid classes"
        )

    return result


def _normalized_yaml(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = dict(payload)
    normalized.pop("path", None)
    return normalized


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0

    try:
        with path.open("rb") as stream:
            while chunk := stream.read(_CHUNK_SIZE):
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise YoloRepeatAuditError(
            f"could not hash file: {path}"
        ) from exc

    return digest.hexdigest(), size


def _tree_counts(
    files: Mapping[str, Path],
) -> dict[str, object]:
    counts: dict[str, object] = {
        "total_regular_files": len(files),
    }

    for modality in _MODALITIES:
        modality_counts: dict[str, object] = {
            "images": {},
            "labels": {},
        }

        for split in _SPLITS:
            image_prefix = (
                f"{modality}/images/{split}/"
            )
            label_prefix = (
                f"{modality}/labels/{split}/"
            )

            images = sum(
                relative.startswith(image_prefix)
                for relative in files
            )
            labels = sum(
                relative.startswith(label_prefix)
                for relative in files
            )

            if images <= 0 or labels <= 0:
                raise YoloRepeatAuditError(
                    f"missing {modality} {split} "
                    "images or labels"
                )

            image_counts = modality_counts["images"]
            label_counts = modality_counts["labels"]

            assert isinstance(image_counts, dict)
            assert isinstance(label_counts, dict)

            image_counts[split] = images
            label_counts[split] = labels

        counts[modality] = modality_counts

    return counts


def audit_yolo_export_repeat(
    first_root: Path,
    repeat_root: Path,
) -> dict[str, object]:
    """Verify two independently generated exports."""

    first = _validated_root(
        first_root,
        label="first",
    )
    repeat = _validated_root(
        repeat_root,
        label="repeat",
    )

    if first == repeat:
        raise YoloRepeatAuditError(
            "first and repeat export roots must differ"
        )

    first_files = _file_map(
        first,
        label="first",
    )
    repeat_files = _file_map(
        repeat,
        label="repeat",
    )

    first_names = set(first_files)
    repeat_names = set(repeat_files)

    if first_names != repeat_names:
        missing = sorted(
            first_names - repeat_names
        )
        extra = sorted(
            repeat_names - first_names
        )
        raise YoloRepeatAuditError(
            "export file trees differ: "
            f"missing={missing[:5]!r}, "
            f"extra={extra[:5]!r}"
        )

    for modality in _MODALITIES:
        first_yaml = _load_yaml(
            first,
            modality=modality,
        )
        repeat_yaml = _load_yaml(
            repeat,
            modality=modality,
        )

        if (
            _normalized_yaml(first_yaml)
            != _normalized_yaml(repeat_yaml)
        ):
            raise YoloRepeatAuditError(
                f"{modality}.yaml structures differ"
            )

    payload_names = sorted(
        first_names - _YAML_FILES
    )
    if not payload_names:
        raise YoloRepeatAuditError(
            "export contains no payload files"
        )

    tree_digest = hashlib.sha256()
    payload_bytes = 0

    for relative in payload_names:
        first_hash, first_size = _sha256_file(
            first_files[relative]
        )
        repeat_hash, repeat_size = _sha256_file(
            repeat_files[relative]
        )

        if (
            first_size != repeat_size
            or first_hash != repeat_hash
        ):
            raise YoloRepeatAuditError(
                f"payload differs: {relative}"
            )

        payload_bytes += first_size
        tree_digest.update(
            relative.encode("utf-8")
        )
        tree_digest.update(b"\0")
        tree_digest.update(
            bytes.fromhex(first_hash)
        )

    first_counts = _tree_counts(first_files)
    repeat_counts = _tree_counts(repeat_files)

    if first_counts != repeat_counts:
        raise YoloRepeatAuditError(
            "export file counts differ"
        )

    return {
        "status": "passed",
        "roots": {
            "first": str(first),
            "repeat": str(repeat),
        },
        "file_counts": {
            "first": first_counts,
            "repeat": repeat_counts,
        },
        "payload_files_compared": len(
            payload_names
        ),
        "payload_bytes_compared": payload_bytes,
        "payload_files_byte_identical": True,
        "payload_tree_sha256": (
            tree_digest.hexdigest()
        ),
        "yaml_structurally_equivalent": True,
        "test_exported": True,
    }