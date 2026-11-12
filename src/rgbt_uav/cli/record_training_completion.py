from __future__ import annotations

import argparse
import json
import math
import os
import re
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

from rgbt_uav.experiments.ledger import (
    ExperimentRecord,
    LedgerError,
    append_record,
)


class CompletionRecordError(ValueError):
    """Raised when completion evidence cannot be safely recorded."""


_ValueErrorT = TypeVar("_ValueErrorT", bound=ValueError)
_REQUIRED_COMPLETION_FIELDS = frozenset(
    {
        "status",
        "experiment_id",
        "experiment_group",
        "config",
        "completed_epochs",
        "pause_checkpoint",
        "pause_checkpoint_sha256",
        "best_weights",
        "best_weights_sha256",
        "last_weights",
        "last_weights_sha256",
        "metrics",
        "elapsed_seconds",
        "timing",
        "peak_cuda_memory_bytes",
        "git_commit",
        "environment",
        "dataset_provenance",
        "dataset_provenance_path",
        "dataset_provenance_sha256",
        "data_yaml_sha256",
        "selected_train_sha256",
        "selected_val_sha256",
        "selection_files",
    }
)


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant: {value}")


class _UnsafeCompletionPathError(OSError):
    """Raised when a no-follow handle is not a regular file handle."""


def _windows_file_attributes(handle: object) -> int:
    import ctypes
    from ctypes import wintypes

    class FileAttributeTagInfo(ctypes.Structure):
        _fields_ = [
            ("file_attributes", wintypes.DWORD),
            ("reparse_tag", wintypes.DWORD),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_information = kernel32.GetFileInformationByHandleEx
    get_information.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    )
    get_information.restype = wintypes.BOOL
    information = FileAttributeTagInfo()
    if not get_information(
        handle,
        9,
        ctypes.byref(information),
        ctypes.sizeof(information),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return information.file_attributes


def _require_windows_non_reparse_handle(
    handle: object,
    *,
    attributes_reader: Callable[[object], int] = _windows_file_attributes,
) -> None:
    if attributes_reader(handle) & 0x00000400:
        raise _UnsafeCompletionPathError("completion handle is a reparse point")


def _open_regular_nofollow_windows(path: Path) -> int:
    import ctypes
    import msvcrt
    from ctypes import wintypes

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
    handle = create_file(
        str(path),
        0x80000000 | 0x00000080,
        0x00000001 | 0x00000002 | 0x00000004,
        None,
        3,
        0x00200000,
        None,
    )
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())

    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    try:
        get_file_type = kernel32.GetFileType
        get_file_type.argtypes = (wintypes.HANDLE,)
        get_file_type.restype = wintypes.DWORD
        if get_file_type(handle) != 0x0001:
            raise _UnsafeCompletionPathError("completion handle is not a disk file")

        _require_windows_non_reparse_handle(handle)
        descriptor = msvcrt.open_osfhandle(
            handle,
            os.O_RDONLY | getattr(os, "O_BINARY", 0),
        )
    except BaseException:
        close_handle(handle)
        raise
    return descriptor


def _open_regular_nofollow(path: Path) -> int:
    if os.name == "nt":
        return _open_regular_nofollow_windows(path)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise OSError("O_NOFOLLOW is unavailable")
    return os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | nofollow)


def _load_json_object(
    path: Path,
    *,
    error_type: type[_ValueErrorT] = CompletionRecordError,
) -> dict[str, object]:
    source = Path(path)

    def invalid(message: str) -> _ValueErrorT:
        return error_type(message)

    try:
        path_stat = source.lstat()
    except OSError as exc:
        raise invalid(
            f"completion must be a readable regular non-symlink file: {source}"
        ) from exc
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
        raise invalid(f"completion must be a regular non-symlink file: {source}")

    def require_path_matches(opened_stat: os.stat_result) -> None:
        try:
            current_stat = source.lstat()
        except OSError as exc:
            raise invalid("completion path is no longer available") from exc
        if (
            stat.S_ISLNK(current_stat.st_mode)
            or not stat.S_ISREG(current_stat.st_mode)
            or not os.path.samestat(current_stat, opened_stat)
        ):
            raise invalid("completion path identity changed or became a symlink")

    try:
        descriptor = _open_regular_nofollow(source)
        with os.fdopen(descriptor, "rb") as stream:
            opened_stat = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened_stat.st_mode) or not os.path.samestat(
                path_stat, opened_stat
            ):
                raise invalid("completion path identity changed while opening")
            require_path_matches(opened_stat)
            content = stream.read()
            ending_stat = os.fstat(stream.fileno())
            if not os.path.samestat(opened_stat, ending_stat):
                raise invalid("completion file identity changed while reading")
            require_path_matches(ending_stat)
    except ValueError:
        raise
    except _UnsafeCompletionPathError as exc:
        raise invalid(f"completion must be a regular non-symlink file: {source}") from exc
    except OSError as exc:
        raise invalid(f"could not read completion file: {source}") from exc

    try:
        decoded = content.decode("utf-8")
        payload = json.loads(decoded, parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise invalid("completion must contain valid UTF-8 strict JSON") from exc
    if not isinstance(payload, dict):
        raise invalid("completion JSON must be an object")
    return payload


def _nonempty_string(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise CompletionRecordError(f"{label} must be a nonempty string")
    return value


def _exact_integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise CompletionRecordError(f"{label} must be an exact integer")
    return value


def _finite_nonnegative(value: object, label: str) -> float:
    if type(value) not in (int, float):
        raise CompletionRecordError(f"{label} must be a finite nonnegative number")
    try:
        converted = float(value)
    except OverflowError as exc:
        raise CompletionRecordError(f"{label} must be a finite nonnegative number") from exc
    if not math.isfinite(converted) or converted < 0:
        raise CompletionRecordError(f"{label} must be a finite nonnegative number")
    return converted


def _hex_digest(value: object, length: int, label: str) -> str:
    if type(value) is not str or re.fullmatch(rf"[0-9a-fA-F]{{{length}}}", value) is None:
        raise CompletionRecordError(f"{label} must be exactly {length} hexadecimal characters")
    return value.lower()


def _verified_metrics(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise CompletionRecordError("metrics must be a nonempty mapping")
    metrics: dict[str, float] = {}
    for key, number in value.items():
        if type(key) is not str or type(number) not in (int, float):
            raise CompletionRecordError("metrics must map strings to finite numbers")
        try:
            converted = float(number)
        except OverflowError as exc:
            raise CompletionRecordError("metrics must map strings to finite numbers") from exc
        if not math.isfinite(converted):
            raise CompletionRecordError("metrics must map strings to finite numbers")
        metrics[key] = converted
    return metrics


def record_completion(completion_path: Path, ledger_path: Path) -> ExperimentRecord:
    completion = _load_json_object(completion_path)
    missing = sorted(_REQUIRED_COMPLETION_FIELDS - completion.keys())
    if missing:
        raise CompletionRecordError(f"completion is missing required fields: {', '.join(missing)}")
    if completion["status"] != "completed":
        raise CompletionRecordError("completion status must be completed")

    experiment_id = _nonempty_string(completion["experiment_id"], "experiment_id")
    experiment_group = _nonempty_string(
        completion["experiment_group"], "experiment_group"
    )
    epochs = completion["completed_epochs"]
    if (
        type(epochs) is not list
        or epochs != [1, 2, 3, 4, 5]
        or any(type(epoch) is not int for epoch in epochs)
    ):
        raise CompletionRecordError("completed_epochs must be exactly [1, 2, 3, 4, 5]")

    config = completion["config"]
    if not isinstance(config, Mapping):
        raise CompletionRecordError("config must be a mapping")
    try:
        config_id = config["experiment_id"]
        config_group = config["experiment_group"]
        seed_value = config["seed"]
        data_value = config["data"]
    except KeyError as exc:
        raise CompletionRecordError(f"config is missing required field: {exc.args[0]}") from exc
    if type(config_id) is not str or config_id != experiment_id:
        raise CompletionRecordError("config experiment_id must exactly match completion")
    if type(config_group) is not str or config_group != experiment_group:
        raise CompletionRecordError("config experiment_group must exactly match completion")
    seed = _exact_integer(seed_value, "config.seed")
    _nonempty_string(data_value, "config.data")

    metrics = _verified_metrics(completion["metrics"])
    elapsed_seconds = _finite_nonnegative(completion["elapsed_seconds"], "elapsed_seconds")
    git_commit = _hex_digest(completion["git_commit"], 40, "git_commit")
    manifest_sha256 = _hex_digest(
        completion["dataset_provenance_sha256"], 64, "dataset_provenance_sha256"
    )
    train_sha256 = _hex_digest(
        completion["selected_train_sha256"], 64, "selected_train_sha256"
    )
    val_sha256 = _hex_digest(
        completion["selected_val_sha256"], 64, "selected_val_sha256"
    )
    _nonempty_string(completion["pause_checkpoint"], "pause_checkpoint")
    pause_sha256 = _hex_digest(
        completion["pause_checkpoint_sha256"], 64, "pause_checkpoint_sha256"
    )

    environment = completion["environment"]
    if not isinstance(environment, Mapping) or not environment:
        raise CompletionRecordError("environment must be a nonempty mapping")
    try:
        hardware = json.dumps(
            environment,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CompletionRecordError("environment must serialize to strict JSON") from exc

    absolute_completion = Path(completion_path).absolute()
    notes = (
        f"selected_train_sha256={train_sha256}; "
        f"selected_val_sha256={val_sha256}; "
        f"pause_checkpoint_sha256={pause_sha256}; resumed epoch 3"
    )
    record = ExperimentRecord(
        experiment_id=experiment_id,
        group=experiment_group,
        timestamp=datetime.now(timezone.utc).isoformat(),
        config_path=f"{absolute_completion}#config",
        git_commit=git_commit,
        manifest_sha256=manifest_sha256,
        seed=seed,
        hardware=hardware,
        metrics=metrics,
        gpu_hours=elapsed_seconds / 3600,
        status="completed",
        notes=notes,
    )
    try:
        append_record(Path(ledger_path), record)
    except LedgerError as exc:
        raise CompletionRecordError(f"could not append completion to ledger: {exc}") from exc
    return record


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record verified training completion evidence")
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, default=Path("runs/ledger.jsonl"))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    record = record_completion(args.completion, args.ledger)
    print(
        json.dumps(
            asdict(record),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
