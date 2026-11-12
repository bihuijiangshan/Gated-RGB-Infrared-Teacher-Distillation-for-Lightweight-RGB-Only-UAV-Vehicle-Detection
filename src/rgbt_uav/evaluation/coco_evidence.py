"""Strict publication of reproducible COCO small-object evaluation evidence."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import math
import os
import platform
import re
import secrets
import stat
import subprocess
import tempfile
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from numbers import Real
from pathlib import Path

from rgbt_uav.evaluation.coco_dataset import (
    CocoDatasetBundle,
    DatasetContractError,
    _is_link_or_reparse,
    _safe_component_observations,
    load_yolo_coco_dataset,
    revalidate_dataset,
)
from rgbt_uav.evaluation.coco_inference import (
    InferenceSettings,
    PredictionEvidence,
    _snapshot_weights,
    collect_coco_predictions,
)
from rgbt_uav.evaluation.coco_metrics import (
    CocoSmallMetrics,
    evaluate_coco_bbox,
)


class EvidencePublicationError(ValueError):
    """Raised when evaluation evidence cannot be validated or published safely."""


@dataclass(frozen=True)
class EvaluationConfig:
    experiment_id: str
    weights: Path
    data: Path
    split: str
    output: Path


@dataclass(frozen=True)
class EvaluationEvidence:
    ground_truth: dict[str, object]
    predictions: tuple[dict[str, object], ...]
    metrics: dict[str, object]
    log_text: str


@dataclass(frozen=True)
class _Identity:
    device: int
    inode: int
    file_type: int
    links: int


@dataclass(frozen=True)
class _OwnedArtifact:
    path: Path
    identity: _Identity
    content: bytes


@dataclass(frozen=True)
class _StagedEntry:
    inode: int
    is_regular: bool
    is_reparse: bool


@dataclass(frozen=True)
class _DirectoryAnchor:
    path: Path
    descriptors: tuple[int, ...]
    delete_guarded: bool = False

    @property
    def leaf_descriptor(self) -> int:
        return self.descriptors[-1]


_SHA256 = re.compile(r"[0-9a-f]{64}\Z", flags=re.ASCII)
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z", flags=re.ASCII)
_PYTHON_VERSION = re.compile(r"([0-9]+)\.([0-9]+)\.([0-9]+)\Z", flags=re.ASCII)
_ENVIRONMENT_KEYS = frozenset({"python", "torch", "ultralytics", "pycocotools"})
_EXPECTED_ENVIRONMENT = {
    "torch": "2.7.1+cu128",
    "ultralytics": "8.3.0",
    "pycocotools": "2.0.11",
}
_WINDOWS_REPARSE_ATTRIBUTE = 0x400
_ARTIFACT_NAMES = (
    "ground_truth.coco.json",
    "predictions.coco.json",
    "metrics.json",
    "evaluation.log",
)
_IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


def _raise(message: str, *, cause: Exception | None = None) -> None:
    error = EvidencePublicationError(message)
    if cause is None:
        raise error
    raise error from cause


def _contains_control(value: str) -> bool:
    return any(unicodedata.category(character) == "Cc" for character in value)


def _absolute(path: object, *, label: str) -> Path:
    try:
        return Path(os.path.abspath(os.fspath(path)))  # type: ignore[arg-type]
    except (OSError, TypeError, ValueError) as exc:
        _raise(f"{label} must be a usable filesystem path", cause=exc)


def _identity(status: os.stat_result) -> _Identity:
    return _Identity(
        status.st_dev,
        status.st_ino,
        stat.S_IFMT(status.st_mode),
        status.st_nlink,
    )


def _lstat(path: Path) -> os.stat_result | None:
    try:
        return os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as exc:
        _raise(f"could not inspect path: {path}", cause=exc)


def _validate_initial_config(config: EvaluationConfig) -> tuple[Path, _Identity]:
    experiment_id = config.experiment_id
    if (
        type(experiment_id) is not str
        or not experiment_id
        or experiment_id != experiment_id.strip()
        or experiment_id in {".", ".."}
        or "/" in experiment_id
        or "\\" in experiment_id
        or _contains_control(experiment_id)
    ):
        _raise("experiment ID must be nonempty, trimmed, and path/control-character free")
    if config.split != "val":
        _raise("split must be the exact value 'val'")

    output = _absolute(config.output, label="output")
    if not output.name:
        _raise("output must name an absent child of an existing parent")
    parent = output.parent
    try:
        observations = _safe_component_observations(parent, label="output parent")
    except DatasetContractError as exc:
        _raise(f"output parent is missing or unsafe: {parent}", cause=exc)
    parent_status = observations[-1][1]
    if not stat.S_ISDIR(parent_status.st_mode) or _is_link_or_reparse(parent_status):
        _raise(f"output parent must be a regular non-reparse directory: {parent}")
    if _lstat(output) is not None:
        _raise(f"output must be absent: {output}")
    return output, _identity(parent_status)


def _require_sequence(value: object, *, label: str) -> list[object] | tuple[object, ...]:
    if type(value) not in {list, tuple}:
        _raise(f"{label} must be a list or tuple")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        _raise(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _validate_results(
    config: EvaluationConfig,
    bundle: CocoDatasetBundle,
    predictions: PredictionEvidence,
    metrics: CocoSmallMetrics,
) -> None:
    images = _require_sequence(bundle.ground_truth.get("images"), label="ground truth images")
    annotations = _require_sequence(
        bundle.ground_truth.get("annotations"), label="ground truth annotations"
    )
    categories = _require_sequence(
        bundle.ground_truth.get("categories"), label="ground truth categories"
    )
    if not images:
        _raise("ground truth images must contain at least one image")
    if not annotations:
        _raise("ground truth annotations must contain at least one annotation")
    expected_categories = [
        {"id": category_id, "name": name}
        for category_id, name in enumerate(bundle.class_names)
    ]
    if list(categories) != expected_categories:
        _raise("ground truth categories must exactly match the ordered bundle classes")
    if len(bundle.class_names) != 5 or tuple(bundle.class_names) != (
        "car",
        "truck",
        "bus",
        "van",
        "freight_car",
    ):
        _raise("bundle classes must be the exact ordered five-class contract")
    if predictions.images_seen != len(bundle.images):
        _raise("inference image count does not equal the dataset image count")
    if predictions.predictions_seen != len(predictions.predictions):
        _raise("inference prediction count does not equal the prediction sequence length")
    if predictions.settings != InferenceSettings():
        _raise("inference settings do not match the official evaluation settings")
    expected_weights = _absolute(config.weights, label="weights")
    evidence_weights = _absolute(predictions.weights_path, label="weights evidence path")
    if os.path.normcase(os.fspath(evidence_weights)) != os.path.normcase(
        os.fspath(expected_weights)
    ):
        _raise("weights evidence path does not match the configured checkpoint")
    _require_sha256(predictions.weights_sha256, label="weights evidence SHA-256")
    _require_sha256(bundle.data_sha256, label="dataset SHA-256")

    expected_metric_names = set(bundle.class_names)
    if set(metrics.per_class_ap_small) != expected_metric_names:
        _raise("per-class metric names must exactly match the bundle classes")
    required_metric_values = (
        metrics.ap,
        metrics.ap50,
        metrics.ap75,
        metrics.ap_small,
        metrics.ar_small,
    )
    for value in required_metric_values:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            _raise("all required and defined per-class metrics must be finite numbers")
        if value < 0.0:
            _raise("required COCO aggregates must not be undefined")
    for value in metrics.per_class_ap_small.values():
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            _raise("all required and defined per-class metrics must be finite numbers")


def _default_environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": metadata.version("torch"),
        "ultralytics": metadata.version("ultralytics"),
        "pycocotools": metadata.version("pycocotools"),
    }


def _default_git_commit() -> str:
    repository_root = _evaluator_project_root()
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tracked_status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=no"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        _raise("could not read evaluator Git provenance", cause=exc)
    if _GIT_COMMIT.fullmatch(head) is None:
        _raise("evaluator Git HEAD must be exactly 40 lowercase hexadecimal characters")
    if tracked_status:
        _raise("evaluator Git tracked working tree must be clean")
    return head


def _evaluator_project_root() -> Path:
    source = Path(__file__).resolve()
    for candidate in source.parents:
        if (candidate / ".git").exists():
            return candidate
    _raise("could not derive evaluator project repository root from installed source")


def _provider_value(provider: Callable[[], object], *, label: str) -> object:
    try:
        return provider()
    except Exception as exc:
        _raise(f"{label} provider failed", cause=exc)


def _validated_environment(provider: Callable[[], object]) -> dict[str, str]:
    supplied = _provider_value(provider, label="environment")
    if not isinstance(supplied, Mapping) or set(supplied) != _ENVIRONMENT_KEYS:
        _raise("environment must contain exactly python, torch, ultralytics, and pycocotools")
    result: dict[str, str] = {}
    for key in ("python", "torch", "ultralytics", "pycocotools"):
        value = supplied[key]
        if (
            type(value) is not str
            or not value
            or value != value.strip()
            or _contains_control(value)
        ):
            _raise(f"environment {key} must be a nonempty control-free string")
        if key == "python":
            match = _PYTHON_VERSION.fullmatch(value)
            if value != platform.python_version():
                _raise("environment python must record the active interpreter version")
            if match is None or not (
                int(match.group(1)) == 3
                and 10 <= int(match.group(2)) < 13
            ):
                _raise("environment python must be within the supported >=3.10,<3.13 range")
        elif value != _EXPECTED_ENVIRONMENT[key]:
            _raise(
                f"environment {key} must exactly match {_EXPECTED_ENVIRONMENT[key]}"
            )
        result[key] = value
    return result


def _validated_time(provider: Callable[[], object]) -> str:
    supplied = _provider_value(provider, label="clock")
    if not isinstance(supplied, datetime) or supplied.tzinfo is None:
        _raise("clock must return a timezone-aware datetime")
    try:
        offset = supplied.utcoffset()
    except (OverflowError, ValueError) as exc:
        _raise("clock returned an invalid timezone-aware datetime", cause=exc)
    if offset is None:
        _raise("clock must return a timezone-aware datetime")
    try:
        return supplied.astimezone(timezone.utc).isoformat()
    except (OverflowError, ValueError) as exc:
        _raise("clock value cannot be represented in UTC", cause=exc)


def _validated_git_commit(provider: Callable[[], object]) -> str:
    supplied = _provider_value(provider, label="git")
    if type(supplied) is not str or _GIT_COMMIT.fullmatch(supplied) is None:
        _raise("git commit must be exactly 40 lowercase hexadecimal characters")
    return supplied


def _json_bytes(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        _raise("evidence is not strict JSON serializable", cause=exc)
    return (text + "\n").encode("utf-8")


def _write_bytes(handle: object, content: bytes) -> object:
    return handle.write(content)  # type: ignore[attr-defined,no-any-return]


def _flush_file(handle: object) -> None:
    handle.flush()  # type: ignore[attr-defined]


def _fsync_file(descriptor: int) -> None:
    os.fsync(descriptor)


def _write_owned(
    path: Path,
    content: bytes,
    owned: list[_OwnedArtifact],
    staging_anchor: _DirectoryAnchor,
    staging_identity: _Identity,
) -> None:
    _require_staging_identity(staging_anchor, staging_identity)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        if os.name == "nt":
            descriptor = os.open(path, flags, 0o600)
        else:
            descriptor = os.open(
                path.name,
                flags,
                0o600,
                dir_fd=staging_anchor.leaf_descriptor,
            )
    except OSError as exc:
        _raise(f"could not exclusively create evidence artifact: {path.name}", cause=exc)
    try:
        opened = os.fstat(descriptor)
    except OSError as exc:
        try:
            os.close(descriptor)
        except OSError:
            pass
        _raise(f"could not inspect created evidence artifact: {path.name}", cause=exc)
    if not stat.S_ISREG(opened.st_mode) or _is_link_or_reparse(opened):
        try:
            os.close(descriptor)
        except OSError:
            pass
        _raise(f"created evidence artifact is not a regular non-reparse file: {path.name}")
    record = _OwnedArtifact(path, _identity(opened), content)
    owned.append(record)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            written = _write_bytes(handle, content)
            if written != len(content):
                raise OSError(
                    f"short evidence artifact write: {written} of {len(content)} bytes"
                )
            _flush_file(handle)
            _fsync_file(handle.fileno())
    except (OSError, ValueError) as exc:
        _raise(f"could not write and sync evidence artifact: {path.name}", cause=exc)
    _require_staging_identity(staging_anchor, staging_identity)


def _read_verified_owned(
    record: _OwnedArtifact,
    staging_anchor: _DirectoryAnchor,
    staging_identity: _Identity,
) -> bytes:
    _require_staging_identity(staging_anchor, staging_identity)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        if os.name == "nt":
            descriptor = os.open(record.path, flags)
        else:
            descriptor = os.open(
                record.path.name,
                flags,
                dir_fd=staging_anchor.leaf_descriptor,
            )
    except OSError as exc:
        _raise(f"could not open staged evidence artifact: {record.path.name}", cause=exc)
    try:
        before = os.fstat(descriptor)
    except OSError as exc:
        try:
            os.close(descriptor)
        except OSError:
            pass
        _raise(f"could not inspect staged evidence artifact: {record.path.name}", cause=exc)
    if (
        _identity(before) != record.identity
        or not stat.S_ISREG(before.st_mode)
        or _is_link_or_reparse(before)
    ):
        try:
            os.close(descriptor)
        except OSError:
            pass
        _raise(f"staged evidence artifact identity changed: {record.path.name}")
    try:
        with os.fdopen(descriptor, "rb") as handle:
            content = handle.read()
            after = os.fstat(handle.fileno())
    except OSError as exc:
        _raise(f"could not read staged evidence artifact: {record.path.name}", cause=exc)
    if _identity(after) != record.identity:
        _raise(f"staged evidence artifact changed while read: {record.path.name}")
    _require_staging_identity(staging_anchor, staging_identity)
    return content


def _sha256_owned(
    record: _OwnedArtifact,
    staging_anchor: _DirectoryAnchor,
    staging_identity: _Identity,
) -> str:
    content = _read_verified_owned(record, staging_anchor, staging_identity)
    if content != record.content:
        _raise(f"staged evidence artifact bytes changed: {record.path.name}")
    return hashlib.sha256(content).hexdigest()


def _windows_staged_directory_entries(
    anchor: _DirectoryAnchor,
) -> dict[str, _StagedEntry]:
    import ctypes
    import msvcrt
    import struct
    from ctypes import wintypes

    get_information = ctypes.WinDLL(
        "kernel32", use_last_error=True
    ).GetFileInformationByHandleEx
    get_information.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    )
    get_information.restype = wintypes.BOOL
    raw_handle = msvcrt.get_osfhandle(anchor.leaf_descriptor)
    buffer = ctypes.create_string_buffer(64 * 1024)
    information_class = 11  # FileIdBothDirectoryRestartInfo
    entries: dict[str, _StagedEntry] = {}
    while True:
        ctypes.set_last_error(0)
        if not get_information(
            raw_handle, information_class, buffer, ctypes.sizeof(buffer)
        ):
            error_number = ctypes.get_last_error()
            if error_number == 18:  # ERROR_NO_MORE_FILES
                break
            _raise(
                "could not enumerate the held Windows staging directory",
                cause=OSError(error_number),
            )
        raw = buffer.raw
        offset = 0
        while True:
            if offset + 104 > len(raw):
                _raise("Windows staging directory returned a malformed entry")
            next_offset = struct.unpack_from("<I", raw, offset)[0]
            attributes = struct.unpack_from("<I", raw, offset + 56)[0]
            name_length = struct.unpack_from("<I", raw, offset + 60)[0]
            file_id = struct.unpack_from("<Q", raw, offset + 96)[0]
            name_end = offset + 104 + name_length
            if name_length % 2 or name_end > len(raw):
                _raise("Windows staging directory returned a malformed name")
            try:
                name = raw[offset + 104 : name_end].decode("utf-16-le")
            except UnicodeError as exc:
                _raise("Windows staging directory returned an invalid name", cause=exc)
            if name not in {".", ".."}:
                if name in entries:
                    _raise("Windows staging directory returned a duplicate name")
                is_directory = bool(attributes & 0x10)
                is_reparse = bool(attributes & _WINDOWS_REPARSE_ATTRIBUTE)
                entries[name] = _StagedEntry(
                    inode=file_id,
                    is_regular=not is_directory and not is_reparse,
                    is_reparse=is_reparse,
                )
            if next_offset == 0:
                break
            if next_offset % 8 or next_offset < 104 or offset + next_offset >= len(raw):
                _raise("Windows staging directory returned an invalid entry offset")
            offset += next_offset
        information_class = 10  # FileIdBothDirectoryInfo resumes enumeration
    return entries


def _staged_directory_entries(anchor: _DirectoryAnchor) -> dict[str, _StagedEntry]:
    _require_anchor_path_identity(anchor, label="staging directory")
    try:
        if os.name == "nt":
            entries = _windows_staged_directory_entries(anchor)
        else:
            names = os.listdir(anchor.leaf_descriptor)
            entries = {
                name: _StagedEntry(
                    inode=status.st_ino,
                    is_regular=stat.S_ISREG(status.st_mode),
                    is_reparse=_is_link_or_reparse(status),
                )
                for name in names
                for status in (
                    os.stat(
                        name,
                        dir_fd=anchor.leaf_descriptor,
                        follow_symlinks=False,
                    ),
                )
            }
    except OSError as exc:
        _raise("could not enumerate the verified staging directory", cause=exc)
    _require_anchor_path_identity(anchor, label="staging directory")
    return entries


def _verify_staged_artifacts(
    owned: list[_OwnedArtifact],
    staging_anchor: _DirectoryAnchor,
    staging_identity: _Identity,
) -> None:
    _require_staging_identity(staging_anchor, staging_identity)
    if [record.path.name for record in owned] != list(_ARTIFACT_NAMES):
        _raise("the staging transaction does not contain exactly the four evidence artifacts")
    entries = _staged_directory_entries(staging_anchor)
    if set(entries) != set(_ARTIFACT_NAMES):
        _raise("the staging directory must contain exactly the four evidence artifacts")
    for record in owned:
        entry = entries[record.path.name]
        if (
            entry.inode != record.identity.inode
            or not entry.is_regular
            or entry.is_reparse
        ):
            _raise(f"staged evidence artifact identity changed: {record.path.name}")
        if _read_verified_owned(record, staging_anchor, staging_identity) != record.content:
            _raise(f"staged evidence artifact bytes changed: {record.path.name}")


def _sync_directory(
    staging_anchor: _DirectoryAnchor, staging_identity: _Identity
) -> None:
    _require_staging_identity(staging_anchor, staging_identity)
    if os.name == "nt":
        return
    try:
        os.fsync(staging_anchor.leaf_descriptor)
    except OSError as exc:
        _raise(
            f"could not sync evidence directory: {staging_anchor.path}", cause=exc
        )
    _require_staging_identity(staging_anchor, staging_identity)


def _same_identity(path: Path, expected: _Identity) -> bool:
    try:
        status = os.stat(path, follow_symlinks=False)
    except OSError:
        return False
    return _identity(status) == expected


def _windows_by_handle_information_type() -> type[object]:
    import ctypes
    from ctypes import wintypes

    class ByHandleFileInformation(ctypes.Structure):
        _fields_ = [
            ("file_attributes", wintypes.DWORD),
            ("creation_time", wintypes.FILETIME),
            ("last_access_time", wintypes.FILETIME),
            ("last_write_time", wintypes.FILETIME),
            ("volume_serial_number", wintypes.DWORD),
            ("file_size_high", wintypes.DWORD),
            ("file_size_low", wintypes.DWORD),
            ("number_of_links", wintypes.DWORD),
            ("file_index_high", wintypes.DWORD),
            ("file_index_low", wintypes.DWORD),
        ]

    return ByHandleFileInformation


def _windows_delete_owned_identity(
    path: Path, expected: _Identity, *, is_directory: bool
) -> str | None:
    import ctypes
    import msvcrt
    from ctypes import wintypes

    ByHandleFileInformation = _windows_by_handle_information_type()

    class FileDispositionInfo(ctypes.Structure):
        _fields_ = [("delete_file", wintypes.BOOLEAN)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    get_information = kernel32.GetFileInformationByHandle
    get_information.argtypes = (wintypes.HANDLE, ctypes.POINTER(ByHandleFileInformation))
    get_information.restype = wintypes.BOOL
    set_information = kernel32.SetFileInformationByHandle
    set_information.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    )
    set_information.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL

    delete_access = 0x00010000
    read_attributes = 0x00000080
    share_all = 0x00000001 | 0x00000002 | 0x00000004
    open_existing = 3
    open_reparse_point = 0x00200000
    backup_semantics = 0x02000000
    file_attribute_normal = 0x00000080
    file_attribute_directory = 0x00000010
    invalid_handle = ctypes.c_void_p(-1).value
    handle = create_file(
        _windows_extended_path(path),
        delete_access | read_attributes,
        share_all,
        None,
        open_existing,
        file_attribute_normal
        | open_reparse_point
        | (backup_semantics if is_directory else 0),
        None,
    )
    if handle == invalid_handle:
        error_number = ctypes.get_last_error()
        if error_number in {2, 3}:
            return None
        return f"{path.name}: could not open owned object for deletion: {error_number}"
    information = ByHandleFileInformation()
    if not get_information(handle, ctypes.byref(information)):
        error_number = ctypes.get_last_error()
        close_handle(handle)
        return f"{path.name}: could not inspect owned deletion handle: {error_number}"
    opened_is_directory = bool(information.file_attributes & file_attribute_directory)
    if (
        information.file_attributes & _WINDOWS_REPARSE_ATTRIBUTE
        or opened_is_directory != is_directory
    ):
        close_handle(handle)
        return None
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY)
    except OSError as exc:
        close_handle(handle)
        return f"{path.name}: could not bind owned deletion handle: {exc}"
    try:
        opened = os.fstat(descriptor)
    except OSError as exc:
        _close_descriptors([descriptor])
        return f"{path.name}: could not inspect bound deletion handle: {exc}"
    if _identity(opened) != expected:
        _close_descriptors([descriptor])
        return None
    raw_handle = msvcrt.get_osfhandle(descriptor)
    disposition = FileDispositionInfo(True)
    if not set_information(
        raw_handle,
        4,
        ctypes.byref(disposition),
        ctypes.sizeof(disposition),
    ):
        error_number = ctypes.get_last_error()
        _close_descriptors([descriptor])
        return f"{path.name}: could not mark owned object for deletion: {error_number}"
    close_errors = _close_descriptors([descriptor])
    if close_errors:
        return f"{path.name}: could not close owned deletion handle: {'; '.join(close_errors)}"
    return None


def _posix_delete_owned_identity(
    path: Path, expected: _Identity, *, is_directory: bool
) -> str | None:
    del expected, is_directory
    return (
        f"{path.name}: retained because POSIX has no atomic identity-bound "
        "deletion primitive"
    )


def _delete_owned_identity(
    path: Path, expected: _Identity, *, is_directory: bool = False
) -> str | None:
    if os.name == "nt":
        return _windows_delete_owned_identity(
            path, expected, is_directory=is_directory
        )
    return _posix_delete_owned_identity(path, expected, is_directory=is_directory)


def _cleanup_owned(
    staging: Path, staging_identity: _Identity, owned: list[_OwnedArtifact]
) -> list[str]:
    if os.name != "nt":
        return [
            f"{staging.name}: retained because POSIX cleanup cannot atomically "
            "bind deletion to the recorded staging identity"
        ]
    errors: list[str] = []
    for record in reversed(owned):
        error = _delete_owned_identity(record.path, record.identity)
        if error is not None:
            errors.append(error)
    staging_error = _delete_owned_identity(
        staging, staging_identity, is_directory=True
    )
    if staging_error is not None:
        errors.append(staging_error)
    return errors


def _revalidate_sources(bundle: CocoDatasetBundle, predictions: PredictionEvidence) -> None:
    try:
        revalidate_dataset(bundle)
    except Exception as exc:
        _raise("dataset revalidation failed immediately before publication", cause=exc)
    weights_path = _absolute(predictions.weights_path, label="weights evidence path")
    try:
        snapshot = _snapshot_weights(weights_path, verified_path=weights_path)
    except Exception as exc:
        _raise("weights revalidation failed immediately before publication", cause=exc)
    if snapshot.fingerprint.sha256 != predictions.weights_sha256:
        _raise("weights SHA-256 changed immediately before publication")


def _confirm_output_absent(
    output: Path, parent_anchor: _DirectoryAnchor
) -> None:
    if output.parent != parent_anchor.path:
        _raise("output is not under the held destination parent")
    _require_anchor_path_identity(parent_anchor, label="destination parent")
    if os.name == "nt":
        appeared = _lstat(output) is not None
    else:
        try:
            os.stat(
                output.name,
                dir_fd=parent_anchor.leaf_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            appeared = False
        except OSError as exc:
            _raise("could not inspect output relative to held destination parent", cause=exc)
        else:
            appeared = True
    if appeared:
        _raise(f"output appeared before publication and will not be replaced: {output}")


def _same_object(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev,
        left.st_ino,
        stat.S_IFMT(left.st_mode),
    ) == (
        right.st_dev,
        right.st_ino,
        stat.S_IFMT(right.st_mode),
    )


def _close_descriptors(descriptors: tuple[int, ...] | list[int]) -> list[str]:
    errors: list[str] = []
    for descriptor in reversed(descriptors):
        try:
            os.close(descriptor)
        except OSError as exc:
            errors.append(str(exc))
    return errors


def _require_anchor_path_identity(anchor: _DirectoryAnchor, *, label: str) -> None:
    try:
        observed = os.stat(anchor.path, follow_symlinks=False)
        opened = os.fstat(anchor.leaf_descriptor)
    except OSError as exc:
        _raise(f"could not revalidate {label} handle against its path", cause=exc)
    if (
        not stat.S_ISDIR(observed.st_mode)
        or _is_link_or_reparse(observed)
        or not _same_object(observed, opened)
    ):
        _raise(f"{label} path identity changed after its handle was verified")


def _require_staging_identity(
    anchor: _DirectoryAnchor, expected: _Identity
) -> None:
    try:
        opened = os.fstat(anchor.leaf_descriptor)
    except OSError as exc:
        _raise("could not inspect the handle-bound staging directory", cause=exc)
    if not stat.S_ISDIR(opened.st_mode) or _identity(opened) != expected:
        _raise("handle-bound staging directory identity differs from its creation identity")
    _require_anchor_path_identity(anchor, label="staging directory")


def _create_staging_directory(
    output: Path, parent_anchor: _DirectoryAnchor
) -> tuple[Path, _Identity]:
    if output.parent != parent_anchor.path:
        _raise("staging output is not under the held destination parent")
    _require_anchor_path_identity(parent_anchor, label="destination parent")
    if os.name == "nt":
        try:
            staging = Path(
                tempfile.mkdtemp(prefix=f".{output.name}.", dir=parent_anchor.path)
            )
        except OSError as exc:
            _raise("could not create a unique sibling evidence staging directory", cause=exc)
        try:
            stage_status = os.stat(staging, follow_symlinks=False)
        except OSError as exc:
            _raise("could not inspect the created staging directory", cause=exc)
    else:
        staging = None
        stage_status = None
        for _ in range(128):
            name = f".{output.name}.{secrets.token_hex(12)}"
            try:
                os.mkdir(name, 0o700, dir_fd=parent_anchor.leaf_descriptor)
            except FileExistsError:
                continue
            except OSError as exc:
                _raise(
                    "could not create staging relative to the held destination parent",
                    cause=exc,
                )
            staging = parent_anchor.path / name
            try:
                stage_status = os.stat(
                    name,
                    dir_fd=parent_anchor.leaf_descriptor,
                    follow_symlinks=False,
                )
            except OSError as exc:
                _raise("could not inspect the created staging directory", cause=exc)
            break
        if staging is None or stage_status is None:
            _raise("could not allocate a unique sibling evidence staging directory")
    _require_anchor_path_identity(parent_anchor, label="destination parent")
    if not stat.S_ISDIR(stage_status.st_mode) or _is_link_or_reparse(stage_status):
        _raise("created staging path is not a regular non-reparse directory")
    return staging, _identity(stage_status)


def _windows_extended_path(path: Path) -> str:
    absolute = str(_absolute(path, label="directory anchor"))
    if absolute.startswith("\\\\?\\"):
        return absolute
    if absolute.startswith("\\\\"):
        return "\\\\?\\UNC\\" + absolute[2:]
    return "\\\\?\\" + absolute


def _open_windows_directory_chain(
    path: Path,
    *,
    leaf_delete_access: bool = False,
    leaf_share_delete: bool = False,
    leaf_list_access: bool = False,
) -> _DirectoryAnchor:
    import ctypes
    import msvcrt
    from ctypes import wintypes

    ByHandleFileInformation = _windows_by_handle_information_type()

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    get_information = kernel32.GetFileInformationByHandle
    get_information.argtypes = (wintypes.HANDLE, ctypes.POINTER(ByHandleFileInformation))
    get_information.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL

    read_attributes = 0x00000080
    list_directory = 0x00000001
    delete_access = 0x00010000
    share_read = 0x00000001
    share_write = 0x00000002
    share_delete = 0x00000004
    open_existing = 3
    open_reparse_point = 0x00200000
    backup_semantics = 0x02000000
    file_attribute_normal = 0x00000080
    file_attribute_directory = 0x00000010
    invalid_handle = ctypes.c_void_p(-1).value
    try:
        observations = _safe_component_observations(path, label="destination parent")
    except DatasetContractError as exc:
        _raise("could not revalidate destination-parent components", cause=exc)
    descriptors: list[int] = []
    leaf_delete_guarded = False
    try:
        for index, (component, expected) in enumerate(observations):
            is_leaf = index == len(observations) - 1
            desired_access = read_attributes | (
                delete_access if is_leaf and leaf_delete_access else 0
            ) | (
                list_directory if is_leaf and leaf_list_access else 0
            )
            share_mode = (
                share_read
                | share_write
                | (share_delete if is_leaf and leaf_share_delete else 0)
            )
            handle = create_file(
                _windows_extended_path(component),
                desired_access,
                share_mode,
                None,
                open_existing,
                file_attribute_normal | open_reparse_point | backup_semantics,
                None,
            )
            used_delete_fallback = False
            if (
                handle == invalid_handle
                and is_leaf
                and leaf_delete_access
                and ctypes.get_last_error() == 32
            ):
                used_delete_fallback = True
                handle = create_file(
                    _windows_extended_path(component),
                    read_attributes,
                    share_mode,
                    None,
                    open_existing,
                    file_attribute_normal | open_reparse_point | backup_semantics,
                    None,
                )
            if handle == invalid_handle:
                _raise(
                    f"could not hold verified destination-parent component: {component}",
                    cause=OSError(ctypes.get_last_error()),
                )
            information = ByHandleFileInformation()
            if not get_information(handle, ctypes.byref(information)):
                error = OSError(ctypes.get_last_error())
                close_handle(handle)
                _raise(
                    f"could not inspect held destination-parent component: {component}",
                    cause=error,
                )
            if (
                information.file_attributes & _WINDOWS_REPARSE_ATTRIBUTE
                or not information.file_attributes & file_attribute_directory
            ):
                close_handle(handle)
                _raise(
                    f"held destination-parent component is reparse or nondirectory: {component}"
                )
            try:
                descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY)
            except OSError as exc:
                close_handle(handle)
                _raise(
                    f"could not bind destination-parent component handle: {component}",
                    cause=exc,
                )
            try:
                opened = os.fstat(descriptor)
            except OSError as exc:
                _close_descriptors([descriptor])
                _raise(
                    f"could not inspect bound destination-parent component: {component}",
                    cause=exc,
                )
            if not stat.S_ISDIR(opened.st_mode) or not _same_object(expected, opened):
                _close_descriptors([descriptor])
                _raise(f"destination-parent component identity changed: {component}")
            descriptors.append(descriptor)
            if is_leaf:
                leaf_delete_guarded = leaf_delete_access and not used_delete_fallback
    except BaseException:
        _close_descriptors(descriptors)
        raise
    return _DirectoryAnchor(path, tuple(descriptors), leaf_delete_guarded)


def _open_posix_directory_chain(path: Path) -> _DirectoryAnchor:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory_flag is None:
        _raise("platform cannot bind destination parent without following links")
    try:
        observations = _safe_component_observations(path, label="destination parent")
    except DatasetContractError as exc:
        _raise("could not revalidate destination-parent components", cause=exc)
    flags = os.O_RDONLY | no_follow | directory_flag | getattr(os, "O_CLOEXEC", 0)
    descriptors: list[int] = []
    try:
        for index, (component, expected) in enumerate(observations):
            try:
                if index == 0:
                    descriptor = os.open(component, flags)
                else:
                    descriptor = os.open(component.name, flags, dir_fd=descriptors[-1])
            except OSError as exc:
                _raise(
                    f"could not hold verified destination-parent component: {component}",
                    cause=exc,
                )
            try:
                opened = os.fstat(descriptor)
            except OSError as exc:
                _close_descriptors([descriptor])
                _raise(
                    f"could not inspect held destination-parent component: {component}",
                    cause=exc,
                )
            if not stat.S_ISDIR(opened.st_mode) or not _same_object(expected, opened):
                _close_descriptors([descriptor])
                _raise(f"destination-parent component identity changed: {component}")
            descriptors.append(descriptor)
    except BaseException:
        _close_descriptors(descriptors)
        raise
    return _DirectoryAnchor(path, tuple(descriptors))


def _open_directory_anchor(
    path: Path,
    *,
    leaf_delete_access: bool = False,
    leaf_share_delete: bool = False,
    leaf_list_access: bool = False,
) -> _DirectoryAnchor:
    if os.name == "nt":
        return _open_windows_directory_chain(
            path,
            leaf_delete_access=leaf_delete_access,
            leaf_share_delete=leaf_share_delete,
            leaf_list_access=leaf_list_access,
        )
    return _open_posix_directory_chain(path)


@contextlib.contextmanager
def _verified_parent_anchor(path: Path):
    anchor = _open_directory_anchor(path, leaf_delete_access=os.name == "nt")
    try:
        yield anchor
    except BaseException as primary:
        errors = _close_descriptors(anchor.descriptors)
        if errors:
            try:
                primary.parent_handle_cleanup_error = "; ".join(errors)
            except Exception:
                pass
        raise
    else:
        _close_descriptors(anchor.descriptors)


@contextlib.contextmanager
def _verified_staging_anchor(path: Path, *, allow_rename: bool = False):
    anchor = _open_directory_anchor(
        path,
        leaf_share_delete=os.name == "nt" and allow_rename,
        leaf_list_access=os.name == "nt",
    )
    try:
        yield anchor
    except BaseException as primary:
        errors = _close_descriptors(anchor.descriptors)
        if errors:
            try:
                primary.staging_handle_cleanup_error = "; ".join(errors)
            except Exception:
                pass
        raise
    else:
        _close_descriptors(anchor.descriptors)


def _invoke_renameat2_noreplace(
    renameat2: object,
    *,
    parent_descriptor: int,
    staging_name: str,
    output_name: str,
    errno_reader: Callable[[], int] | None = None,
) -> None:
    import ctypes

    renameat2.argtypes = (  # type: ignore[attr-defined]
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int  # type: ignore[attr-defined]
    result = renameat2(  # type: ignore[operator]
        parent_descriptor,
        os.fsencode(staging_name),
        parent_descriptor,
        os.fsencode(output_name),
        1,
    )
    if result != 0:
        error_number = (errno_reader or ctypes.get_errno)()
        _raise(
            f"could not publish evidence with renameat2 no-replace: {output_name}",
            cause=OSError(error_number, os.strerror(error_number)),
        )


def _publish_no_replace(
    staging: Path,
    output: Path,
    parent_anchor: _DirectoryAnchor,
    staging_anchor: _DirectoryAnchor,
    staging_identity: _Identity,
) -> None:
    if staging.parent != output.parent or parent_anchor.path != output.parent:
        _raise("staging, output, and verified destination parent must share one directory")
    _require_anchor_path_identity(parent_anchor, label="destination parent")
    _require_staging_identity(staging_anchor, staging_identity)
    if os.name == "nt":
        try:
            opened_staging = os.fstat(staging_anchor.leaf_descriptor)
        except OSError as exc:
            _raise("could not inspect handle-bound staging directory", cause=exc)
        if not stat.S_ISDIR(opened_staging.st_mode):
            _raise("handle-bound staging object is not a directory")
        try:
            os.rename(staging, output)
        except OSError as exc:
            _raise(
                f"could not publish evidence without replacing the output: {output}",
                cause=exc,
            )
        return

    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        _raise("platform does not provide renameat2 no-replace publication")
    _invoke_renameat2_noreplace(
        renameat2,
        parent_descriptor=parent_anchor.leaf_descriptor,
        staging_name=staging.name,
        output_name=output.name,
    )


def _confirm_published(
    output: Path,
    staging_identity: _Identity,
    parent_anchor: _DirectoryAnchor,
) -> _DirectoryAnchor:
    if output.parent != parent_anchor.path:
        _raise("published output is not under the verified destination parent")
    _require_anchor_path_identity(parent_anchor, label="destination parent")
    anchor = _open_directory_anchor(
        output, leaf_delete_access=os.name == "nt"
    )
    try:
        status = os.fstat(anchor.leaf_descriptor)
    except OSError as exc:
        _close_descriptors(anchor.descriptors)
        _raise("could not confirm the published output identity", cause=exc)
    if _identity(status) != staging_identity or not stat.S_ISDIR(status.st_mode):
        _close_descriptors(anchor.descriptors)
        _raise("published output identity does not match the staging directory")
    return anchor


def _finish_committed_publication(
    output: Path,
    parent_anchor: _DirectoryAnchor,
    published_anchor: _DirectoryAnchor | None,
) -> None:
    del output
    try:
        if os.name != "nt":
            os.fsync(parent_anchor.leaf_descriptor)
    finally:
        if published_anchor is not None:
            errors = _close_descriptors(published_anchor.descriptors)
            if errors:
                raise OSError("; ".join(errors))


def _metrics_document(
    config: EvaluationConfig,
    bundle: CocoDatasetBundle,
    predictions: PredictionEvidence,
    metrics: CocoSmallMetrics,
    *,
    environment: dict[str, str],
    git_commit: str,
    verified_at_utc: str,
    ground_truth_sha256: str,
    predictions_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "evaluator_version": 1,
        "status": "passed",
        "experiment_id": config.experiment_id,
        "weights": {
            "path": str(_absolute(predictions.weights_path, label="weights evidence path")),
            "sha256": predictions.weights_sha256,
        },
        "data": {
            "path": str(_absolute(bundle.data_path, label="dataset evidence path")),
            "sha256": bundle.data_sha256,
            "split": "val",
        },
        "git_commit": git_commit,
        "verified_at_utc": verified_at_utc,
        "environment": environment,
        "counts": {
            "images": len(bundle.images),
            "annotations": len(bundle.ground_truth["annotations"]),
            "predictions": predictions.predictions_seen,
        },
        "classes": [
            {"id": category_id, "name": name}
            for category_id, name in enumerate(bundle.class_names)
        ],
        "inference": {
            "imgsz": predictions.settings.imgsz,
            "conf": predictions.settings.conf,
            "iou": predictions.settings.iou,
            "max_det": predictions.settings.max_det,
        },
        "coco": {
            "iou_thresholds": list(_IOU_THRESHOLDS),
            "small_area_range": [0, 1024],
            "max_dets": [1, 10, 100],
            "reported_max_det": 100,
            "bbox_coordinate_space": [640, 512],
        },
        "metrics": {
            "AP": float(metrics.ap),
            "AP50": float(metrics.ap50),
            "AP75": float(metrics.ap75),
            "AP_small": float(metrics.ap_small),
            "AR_small": float(metrics.ar_small),
            "per_class_AP_small": {
                name: (
                    None
                    if metrics.per_class_ap_small[name] is None
                    else float(metrics.per_class_ap_small[name])
                )
                for name in bundle.class_names
            },
        },
        "artifacts": {
            "ground_truth": {
                "file": "ground_truth.coco.json",
                "sha256": ground_truth_sha256,
            },
            "predictions": {
                "file": "predictions.coco.json",
                "sha256": predictions_sha256,
            },
        },
    }


def run_coco_small_evaluation(
    config: EvaluationConfig,
    *,
    dataset_loader: Callable[..., CocoDatasetBundle] = load_yolo_coco_dataset,
    prediction_collector: Callable[..., PredictionEvidence] = collect_coco_predictions,
    metric_evaluator: Callable[..., CocoSmallMetrics] = evaluate_coco_bbox,
    clock: Callable[[], object] | None = None,
    environment_provider: Callable[[], object] | None = None,
    git_commit_provider: Callable[[], object] | None = None,
) -> Path:
    """Publish evidence and return output/metrics.json."""
    output, validated_parent_identity = _validate_initial_config(config)
    with _verified_parent_anchor(output.parent) as parent_anchor:
        try:
            held_parent = _identity(os.fstat(parent_anchor.leaf_descriptor))
        except OSError as exc:
            _raise("could not inspect the held destination parent", cause=exc)
        if (
            held_parent.device,
            held_parent.inode,
            held_parent.file_type,
        ) != (
            validated_parent_identity.device,
            validated_parent_identity.inode,
            validated_parent_identity.file_type,
        ):
            _raise("destination parent identity changed after initial validation")
        _confirm_output_absent(output, parent_anchor)
        bundle = dataset_loader(config.data, split=config.split)
        predictions = prediction_collector(config.weights, bundle)
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            metrics = metric_evaluator(bundle.ground_truth, predictions.predictions)
        log_text = captured.getvalue()
        _require_anchor_path_identity(parent_anchor, label="destination parent")
        _validate_results(config, bundle, predictions, metrics)
        try:
            revalidate_dataset(bundle)
        except Exception as exc:
            _raise("dataset revalidation failed after inference and metrics", cause=exc)

        verified_at_utc = _validated_time(clock or (lambda: datetime.now(timezone.utc)))
        environment = _validated_environment(environment_provider or _default_environment)
        git_commit = _validated_git_commit(git_commit_provider or _default_git_commit)
        ground_truth = copy.deepcopy(bundle.ground_truth)
        prediction_rows = copy.deepcopy(tuple(predictions.predictions))
        ground_truth_bytes = _json_bytes(ground_truth)
        prediction_bytes = _json_bytes(prediction_rows)
        _require_anchor_path_identity(parent_anchor, label="destination parent")

        staging: Path | None = None
        staging_identity: _Identity | None = None
        owned: list[_OwnedArtifact] = []
        committed = False
        try:
            staging, staging_identity = _create_staging_directory(output, parent_anchor)
            with _verified_staging_anchor(staging) as staging_anchor:
                _require_staging_identity(staging_anchor, staging_identity)
                _write_owned(
                    staging / "ground_truth.coco.json",
                    ground_truth_bytes,
                    owned,
                    staging_anchor,
                    staging_identity,
                )
                _write_owned(
                    staging / "predictions.coco.json",
                    prediction_bytes,
                    owned,
                    staging_anchor,
                    staging_identity,
                )
                ground_truth_sha256 = _sha256_owned(
                    owned[0], staging_anchor, staging_identity
                )
                predictions_sha256 = _sha256_owned(
                    owned[1], staging_anchor, staging_identity
                )
                document = _metrics_document(
                    config,
                    bundle,
                    predictions,
                    metrics,
                    environment=environment,
                    git_commit=git_commit,
                    verified_at_utc=verified_at_utc,
                    ground_truth_sha256=ground_truth_sha256,
                    predictions_sha256=predictions_sha256,
                )
                evidence = EvaluationEvidence(
                    ground_truth, prediction_rows, document, log_text
                )
                _write_owned(
                    staging / "metrics.json",
                    _json_bytes(evidence.metrics),
                    owned,
                    staging_anchor,
                    staging_identity,
                )
                _write_owned(
                    staging / "evaluation.log",
                    evidence.log_text.encode("utf-8"),
                    owned,
                    staging_anchor,
                    staging_identity,
                )
                _sync_directory(staging_anchor, staging_identity)
                _verify_staged_artifacts(owned, staging_anchor, staging_identity)

            with _verified_staging_anchor(staging, allow_rename=True) as staging_anchor:
                _require_staging_identity(staging_anchor, staging_identity)
                _verify_staged_artifacts(owned, staging_anchor, staging_identity)
                _revalidate_sources(bundle, predictions)
                _confirm_output_absent(output, parent_anchor)
                _publish_no_replace(
                    staging,
                    output,
                    parent_anchor,
                    staging_anchor,
                    staging_identity,
                )
                committed = True
                try:
                    published_anchor = _confirm_published(
                        output, staging_identity, parent_anchor
                    )
                    _finish_committed_publication(
                        output, parent_anchor, published_anchor
                    )
                except BaseException:
                    pass
        except BaseException as primary:
            if staging is not None and staging_identity is not None and not committed:
                try:
                    cleanup_errors = _cleanup_owned(staging, staging_identity, owned)
                except BaseException as cleanup_error:
                    cleanup_errors = [str(cleanup_error)]
                if cleanup_errors:
                    try:
                        primary.evidence_cleanup_error = "; ".join(cleanup_errors)
                    except Exception:
                        pass
            if isinstance(primary, EvidencePublicationError) or not isinstance(
                primary, Exception
            ):
                raise
            raise EvidencePublicationError(
                "evidence publication transaction failed"
            ) from primary

        return output / "metrics.json"


__all__ = [
    "EvaluationConfig",
    "EvaluationEvidence",
    "EvidencePublicationError",
    "run_coco_small_evaluation",
]
