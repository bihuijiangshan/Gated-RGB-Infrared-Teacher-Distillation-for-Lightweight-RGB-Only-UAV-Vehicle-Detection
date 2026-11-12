"""Strict configuration and durable state for resumable training preflights."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal

import yaml
from yaml.nodes import MappingNode, ScalarNode

from rgbt_uav.training.smoke import (
    environment_snapshot,
    git_provenance,
    peak_cuda_memory_bytes,
    read_dataset_provenance,
    read_finite_metrics,
    write_json_atomic,
)

_CONFIG_KEYS = frozenset(
    {
        "experiment_id",
        "experiment_group",
        "model",
        "data",
        "prediction_source",
        "output_root",
        "epochs",
        "pause_after_epoch",
        "imgsz",
        "rect",
        "batch",
        "seed",
        "workers",
        "device",
        "amp",
        "cache",
    }
)
_BOOLEAN_KEYS = frozenset({"rect", "amp", "cache"})
_INTEGER_KEYS = frozenset({"epochs", "imgsz", "batch", "seed", "workers"})
_STRING_KEYS = frozenset(
    {"experiment_id", "experiment_group", "model", "data", "prediction_source", "output_root"}
)
_STATUSES = frozenset({"running", "paused", "completed", "failed"})
_PHASES = frozenset({"initial", "resume", "finalize"})


class ResumableTrainingError(RuntimeError):
    """Raised when resumable training configuration or state is unsafe."""


class ControlledTrainingPause(RuntimeError):
    """Raised after a requested epoch has been saved so training can resume later."""


@dataclass(frozen=True)
class ResumableCheckpoint:
    """Evidence that a local checkpoint can resume the expected training run."""

    human_epoch: int
    target_epochs: int
    has_optimizer: bool
    sha256: str


@dataclass(frozen=True)
class EpochTiming:
    human_epoch: int
    phase: Literal["initial", "resume"]
    train_seconds: float
    val_seconds: float


@dataclass(frozen=True)
class TrainingRunResult:
    best_weights: Path
    last_weights: Path
    pause_checkpoint: Path
    state_path: Path
    completion_path: Path
    metrics: dict[str, float]
    completed_epochs: list[int]
    epoch_timings: list[EpochTiming]


class _TimingTracker:
    """Pair callback intervals without allowing ambiguous epoch evidence."""

    def __init__(self, *, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._active: dict[str, tuple[int, Literal["initial", "resume"], float]] = {}
        self._train: dict[int, tuple[Literal["initial", "resume"], float]] = {}
        self._val: dict[int, tuple[Literal["initial", "resume"], float]] = {}
        self.framework_final_validation_seconds: float | None = None
        self._sealed = False
        self._latest_train_epoch: tuple[int, Literal["initial", "resume"]] | None = None

    @staticmethod
    def _human_epoch(trainer: object) -> int:
        raw_epoch = getattr(trainer, "epoch", None)
        if type(raw_epoch) is not int or raw_epoch < 0:
            raise ResumableTrainingError("trainer epoch must be a non-negative integer")
        return raw_epoch + 1

    @staticmethod
    def _require_phase(phase: str) -> Literal["initial", "resume"]:
        if phase not in {"initial", "resume"}:
            raise ResumableTrainingError("timing phase must be initial or resume")
        return phase  # type: ignore[return-value]

    def _start(self, kind: str, trainer: object, phase: str) -> None:
        if self._sealed:
            return
        checked_phase = self._require_phase(phase)
        if kind == "train":
            human_epoch = self._human_epoch(trainer)
            self._latest_train_epoch = (human_epoch, checked_phase)
        else:
            if self._latest_train_epoch is None:
                raise ResumableTrainingError("validation interval has no training epoch")
            human_epoch, train_phase = self._latest_train_epoch
            if train_phase != checked_phase:
                raise ResumableTrainingError("validation interval phase changed")
        if kind in self._active:
            raise ResumableTrainingError(f"duplicate {kind} interval start")
        started_at = self._clock()
        if not math.isfinite(started_at):
            raise ResumableTrainingError(f"{kind} interval start must be finite")
        self._active[kind] = (human_epoch, checked_phase, started_at)

    def _end(self, kind: str, trainer: object, phase: str) -> None:
        if self._sealed:
            return
        checked_phase = self._require_phase(phase)
        active = self._active.pop(kind, None)
        if active is None:
            raise ResumableTrainingError(f"missing {kind} interval start")
        active_epoch, active_phase, started_at = active
        human_epoch = self._human_epoch(trainer) if kind == "train" else active_epoch
        if checked_phase != active_phase or human_epoch != active_epoch:
            raise ResumableTrainingError(f"{kind} interval changed epoch or phase")
        ended_at = self._clock()
        duration = ended_at - started_at
        if not math.isfinite(ended_at) or not math.isfinite(duration) or duration < 0:
            raise ResumableTrainingError(f"{kind} interval must be finite and non-negative")

        destination = self._train if kind == "train" else self._val
        if human_epoch not in destination:
            destination[human_epoch] = (checked_phase, duration)
            return
        if (
            kind == "val"
            and human_epoch == 5
            and checked_phase == "resume"
            and self.framework_final_validation_seconds is None
        ):
            self.framework_final_validation_seconds = duration
            return
        raise ResumableTrainingError(f"duplicate {kind} interval for epoch {human_epoch}")

    def train_start(self, trainer: object, phase: str) -> None:
        self._start("train", trainer, phase)

    def train_end(self, trainer: object, phase: str) -> None:
        self._end("train", trainer, phase)

    def val_start(self, trainer: object, phase: str) -> None:
        self._start("val", trainer, phase)

    def val_end(self, trainer: object, phase: str) -> None:
        self._end("val", trainer, phase)

    def completed_epochs(self) -> list[int]:
        return sorted(set(self._train) & set(self._val))

    def epoch_timings(self) -> list[EpochTiming]:
        if self._active:
            raise ResumableTrainingError("timing callbacks left an interval incomplete")
        if set(self._train) != set(self._val):
            raise ResumableTrainingError("each epoch requires train and validation timing")
        timings: list[EpochTiming] = []
        for human_epoch in sorted(self._train):
            train_phase, train_seconds = self._train[human_epoch]
            val_phase, val_seconds = self._val[human_epoch]
            if train_phase != val_phase:
                raise ResumableTrainingError("epoch train and validation phases must match")
            timings.append(
                EpochTiming(
                    human_epoch=human_epoch,
                    phase=train_phase,
                    train_seconds=train_seconds,
                    val_seconds=val_seconds,
                )
            )
        self._sealed = True
        return timings


def _attach_timing_callbacks(
    model: object, tracker: _TimingTracker, phase: Literal["initial", "resume"]
) -> None:
    """Attach the four Ultralytics epoch timing callbacks."""
    add_callback = getattr(model, "add_callback", None)
    if not callable(add_callback):
        raise ResumableTrainingError("training model does not support callbacks")
    add_callback("on_train_epoch_start", lambda trainer: tracker.train_start(trainer, phase))
    add_callback("on_train_epoch_end", lambda trainer: tracker.train_end(trainer, phase))
    add_callback("on_val_start", lambda trainer: tracker.val_start(trainer, phase))
    add_callback("on_val_end", lambda trainer: tracker.val_end(trainer, phase))


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_config_path(declared: str, *, repository: Path) -> Path:
    path = Path(declared)
    if not path.is_absolute():
        path = repository / path
    return path.resolve(strict=False)


def _is_strict_child(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path != root


def _repository_relative(path: Path, *, label: str) -> str:
    repository = _repository_root()
    try:
        return path.resolve(strict=False).relative_to(repository).as_posix()
    except ValueError as exc:
        raise ResumableTrainingError(f"{label} must be within the repository") from exc


def _has_test_component(path: Path) -> bool:
    return any(component.lower() == "test" for component in path.parts)


def _require_non_empty_string(value: object, *, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ResumableTrainingError(f"{label} must be a non-empty string")
    return value


def _require_integer(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise ResumableTrainingError(f"{label} must be an integer")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise ResumableTrainingError(f"{label} must be a SHA-256 hex digest")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ResumableTrainingError(f"could not hash checkpoint: {path}") from exc
    return digest.hexdigest()


def _sha256_open_file(handle: BinaryIO) -> str:
    position = handle.tell()
    digest = hashlib.sha256()
    handle.seek(0)
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    handle.seek(position)
    return digest.hexdigest()


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _has_symlink_component(path: Path) -> bool:
    absolute = _absolute_without_resolving(path)
    return any(component.is_symlink() for component in (absolute, *absolute.parents))


def _validated_checkpoint_path(path: Path, *, allowed_root: Path) -> Path:
    candidate = Path(path)
    root = Path(allowed_root)
    lexical_candidate = _absolute_without_resolving(candidate)
    lexical_root = _absolute_without_resolving(root)
    if not _is_strict_child(lexical_candidate, lexical_root):
        raise ResumableTrainingError("checkpoint must be a strict child of allowed_root")
    if _has_symlink_component(root) or _has_symlink_component(candidate):
        raise ResumableTrainingError("checkpoint path must not contain symlinks")
    try:
        resolved_root = root.resolve(strict=True)
        resolved_candidate = candidate.resolve(strict=True)
    except OSError as exc:
        raise ResumableTrainingError("checkpoint and allowed_root must exist") from exc
    if not resolved_root.is_dir():
        raise ResumableTrainingError("allowed_root must be a directory")
    if not _is_strict_child(resolved_candidate, resolved_root):
        raise ResumableTrainingError("checkpoint must be a strict child of allowed_root")
    if not resolved_candidate.is_file():
        raise ResumableTrainingError("checkpoint must be a regular file")
    return resolved_candidate


def _checkpoint_target_epochs(payload: Mapping[object, object]) -> object:
    for key in ("train_args", "args"):
        arguments = payload.get(key)
        if isinstance(arguments, Mapping) and "epochs" in arguments:
            return arguments["epochs"]
        if arguments is not None and hasattr(arguments, "epochs"):
            return arguments.epochs
    raise ResumableTrainingError("checkpoint does not declare its target epochs")


def _require_path_matches_open_file(
    path: Path, open_file_stat: os.stat_result, *, label: str
) -> os.stat_result:
    if path.is_symlink():
        raise ResumableTrainingError(f"{label} must not be a symlink")
    try:
        path_stat = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise ResumableTrainingError(f"{label} path is no longer available") from exc
    if not stat.S_ISREG(path_stat.st_mode) or not os.path.samestat(path_stat, open_file_stat):
        raise ResumableTrainingError(f"{label} path identity changed")
    return path_stat


def _mark_open_file_for_deletion_windows(file_descriptor: int) -> None:
    import ctypes
    import msvcrt
    from ctypes import wintypes

    class FileDispositionInfo(ctypes.Structure):
        _fields_ = [("delete_file", wintypes.BOOL)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    set_information = kernel32.SetFileInformationByHandle
    set_information.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    )
    set_information.restype = wintypes.BOOL
    disposition = FileDispositionInfo(True)
    if not set_information(
        msvcrt.get_osfhandle(file_descriptor),
        4,
        ctypes.byref(disposition),
        ctypes.sizeof(disposition),
    ):
        raise ctypes.WinError(ctypes.get_last_error())


def _unlink_path_if_owned(path: Path, owned_stat: os.stat_result) -> None:
    if os.name == "nt":
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
            0x00010000 | 0x00000080,
            0x00000001 | 0x00000002 | 0x00000004,
            None,
            3,
            0x00200000,
            None,
        )
        if handle == wintypes.HANDLE(-1).value:
            error_code = ctypes.get_last_error()
            if error_code in {2, 3}:
                return
            raise ctypes.WinError(error_code)
        file_descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY)
        try:
            opened_stat = os.fstat(file_descriptor)
            if not os.path.samestat(opened_stat, owned_stat):
                return
            _mark_open_file_for_deletion_windows(file_descriptor)
        finally:
            os.close(file_descriptor)
        return

    try:
        current_stat = path.lstat()
    except FileNotFoundError:
        return
    except OSError:
        return
    if (current_stat.st_dev, current_stat.st_ino) != (owned_stat.st_dev, owned_stat.st_ino):
        return
    # POSIX exposes no stdlib unlink-by-handle operation. The runner holds its
    # exclusive output-root lock for this fallback's full lifetime; cooperating
    # writers must honor that single-writer contract.
    path.unlink()


def _replace_owned_path_windows(
    source: Path, destination: Path, previous_token: _JsonPublicationToken
) -> None:
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
        str(destination),
        0x80000000 | 0x00010000 | 0x00000080,
        0x00000001 | 0x00000004,
        None,
        3,
        0x00200000,
        None,
    )
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    file_descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY)
    try:
        opened_stat = os.fstat(file_descriptor)
        if not os.path.samestat(opened_stat, previous_token.owned_stat):
            raise ResumableTrainingError("previous run state identity changed")
        _require_single_link(opened_stat, label="previous run state")
        _require_path_matches_open_file(
            destination,
            opened_stat,
            label="previous run state",
        )
        with os.fdopen(os.dup(file_descriptor), "rb") as previous_handle:
            if previous_handle.read() != previous_token.expected_bytes:
                raise ResumableTrainingError("previous run state bytes changed")
        ending_stat = os.fstat(file_descriptor)
        if not os.path.samestat(opened_stat, ending_stat):
            raise ResumableTrainingError("previous run state identity changed")
        _require_single_link(ending_stat, label="previous run state")
        _require_path_matches_open_file(
            destination,
            ending_stat,
            label="previous run state",
        )
        replace_file = kernel32.ReplaceFileW
        replace_file.argtypes = (
            wintypes.LPCWSTR,
            wintypes.LPCWSTR,
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.LPVOID,
        )
        replace_file.restype = wintypes.BOOL
        if not replace_file(
            str(destination),
            str(source),
            None,
            0x00000001,
            None,
            None,
        ):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        os.close(file_descriptor)


def _replace_owned_json_path(
    source: Path, destination: Path, previous_token: _JsonPublicationToken
) -> None:
    if os.name == "nt":
        _replace_owned_path_windows(source, destination, previous_token)
        return
    _reverify_owned_json_publication(previous_token)
    os.replace(source, destination)


def _publish_checkpoint_no_clobber(
    temporary_path: Path,
    destination_path: Path,
    *,
    temporary_handle: BinaryIO,
    temporary_stat: os.stat_result,
    expected_size: int,
    expected_sha256: str,
) -> None:
    try:
        os.link(temporary_path, destination_path, follow_symlinks=False)
    except FileExistsError as exc:
        raise ResumableTrainingError(
            "destination checkpoint appeared during preservation"
        ) from exc
    except OSError as exc:
        raise ResumableTrainingError("could not atomically publish resume checkpoint") from exc

    try:
        published_stat = _require_path_matches_open_file(
            destination_path,
            temporary_stat,
            label="published checkpoint",
        )
        if (
            published_stat.st_size != expected_size
            or _sha256_open_file(temporary_handle) != expected_sha256
        ):
            raise ResumableTrainingError("published checkpoint content changed")
    except BaseException:
        try:
            _unlink_path_if_owned(destination_path, temporary_stat)
        except OSError:
            pass
        raise


def _load_trusted_local_checkpoint(path: Path) -> object:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def inspect_resumable_checkpoint(
    path: Path,
    *,
    allowed_root: Path,
    expected_target_epochs: int,
    expected_human_epoch: int,
    loader: Callable[[Path], object] | None = None,
) -> ResumableCheckpoint:
    """Validate evidence in a trusted checkpoint created beneath ``allowed_root``."""
    _require_integer(expected_target_epochs, label="expected_target_epochs")
    _require_integer(expected_human_epoch, label="expected_human_epoch")
    if expected_target_epochs < 1 or expected_human_epoch < 1:
        raise ResumableTrainingError("expected checkpoint epochs must be positive")

    checkpoint_path = _validated_checkpoint_path(path, allowed_root=allowed_root)
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    checkpoint_loader = loader if loader is not None else _load_trusted_local_checkpoint
    try:
        payload = checkpoint_loader(checkpoint_path)
    except Exception as exc:
        raise ResumableTrainingError("could not load resumable checkpoint") from exc
    if not isinstance(payload, Mapping):
        raise ResumableTrainingError("checkpoint payload must be a mapping")

    raw_epoch = payload.get("epoch")
    if type(raw_epoch) is not int:
        raise ResumableTrainingError("checkpoint epoch must be an integer")
    human_epoch = raw_epoch + 1
    if human_epoch != expected_human_epoch:
        raise ResumableTrainingError(
            f"checkpoint is for human epoch {human_epoch}, expected {expected_human_epoch}"
        )

    target_epochs = _checkpoint_target_epochs(payload)
    if type(target_epochs) is not int:
        raise ResumableTrainingError("checkpoint target epochs must be an integer")
    if target_epochs != expected_target_epochs:
        raise ResumableTrainingError(
            f"checkpoint targets {target_epochs} epochs, expected {expected_target_epochs}"
        )
    if payload.get("optimizer") is None:
        raise ResumableTrainingError("checkpoint optimizer state is missing")

    return ResumableCheckpoint(
        human_epoch=human_epoch,
        target_epochs=target_epochs,
        has_optimizer=True,
        sha256=checkpoint_sha256,
    )


def preserve_resume_checkpoint(source: Path, destination: Path) -> Path:
    """Copy a checkpoint through a verified temporary sibling and atomically publish it."""
    source_path = Path(source)
    destination_path = Path(destination)
    if _has_symlink_component(source_path):
        raise ResumableTrainingError("source checkpoint path must not contain symlinks")
    try:
        resolved_source = source_path.resolve(strict=True)
    except OSError as exc:
        raise ResumableTrainingError("source checkpoint must exist") from exc
    if not resolved_source.is_file():
        raise ResumableTrainingError("source checkpoint must be a regular file")

    destination_parent = destination_path.parent
    if _has_symlink_component(destination_parent):
        raise ResumableTrainingError("destination path must not contain symlinks")
    try:
        resolved_parent = destination_parent.resolve(strict=True)
    except OSError as exc:
        raise ResumableTrainingError("destination parent must exist") from exc
    if not resolved_parent.is_dir():
        raise ResumableTrainingError("destination parent must be a directory")
    if destination_path.exists() or destination_path.is_symlink():
        raise ResumableTrainingError("destination checkpoint must not already exist")

    temporary_path = destination_path.with_suffix(f"{destination_path.suffix}.tmp")
    if temporary_path.exists() or temporary_path.is_symlink():
        raise ResumableTrainingError("temporary checkpoint must not already exist")

    temporary_created = False
    temporary_stat: os.stat_result | None = None
    failed = False
    try:
        try:
            with resolved_source.open("rb") as source_handle, temporary_path.open(
                "x+b"
            ) as temporary_handle:
                temporary_created = True
                source_stat = os.fstat(source_handle.fileno())
                temporary_stat = os.fstat(temporary_handle.fileno())
                _require_path_matches_open_file(
                    resolved_source,
                    source_stat,
                    label="source checkpoint",
                )
                source_size = source_stat.st_size
                source_sha256 = _sha256_open_file(source_handle)

                shutil.copyfileobj(source_handle, temporary_handle)
                temporary_handle.flush()
                os.fsync(temporary_handle.fileno())

                copied_stat = os.fstat(temporary_handle.fileno())
                copied_sha256 = _sha256_open_file(temporary_handle)
                current_source_stat = os.fstat(source_handle.fileno())
                current_source_sha256 = _sha256_open_file(source_handle)
                _require_path_matches_open_file(
                    resolved_source,
                    source_stat,
                    label="source checkpoint",
                )
                if (
                    copied_stat.st_size != source_size
                    or copied_sha256 != source_sha256
                    or current_source_stat.st_size != source_size
                    or current_source_sha256 != source_sha256
                ):
                    raise ResumableTrainingError(
                        "preserved checkpoint does not match its source"
                    )
                _require_path_matches_open_file(
                    temporary_path,
                    temporary_stat,
                    label="temporary checkpoint",
                )
                _publish_checkpoint_no_clobber(
                    temporary_path,
                    destination_path,
                    temporary_handle=temporary_handle,
                    temporary_stat=temporary_stat,
                    expected_size=source_size,
                    expected_sha256=source_sha256,
                )
        except FileExistsError as exc:
            raise ResumableTrainingError("temporary checkpoint must not already exist") from exc
        except OSError as exc:
            raise ResumableTrainingError("could not preserve resume checkpoint") from exc
    except BaseException:
        failed = True
        raise
    finally:
        if temporary_created and temporary_stat is not None:
            try:
                _unlink_path_if_owned(temporary_path, temporary_stat)
            except BaseException as cleanup_exc:
                if not failed:
                    if isinstance(cleanup_exc, OSError):
                        raise ResumableTrainingError(
                            "could not clean temporary checkpoint"
                        ) from cleanup_exc
                    raise

    return destination_path


def make_pause_callback(*, human_epoch: int) -> Callable[[object], None]:
    """Return an Ultralytics callback that pauses after the requested saved epoch."""
    _require_integer(human_epoch, label="human_epoch")
    if human_epoch < 1:
        raise ResumableTrainingError("human_epoch must be positive")

    def pause_after_save(trainer: object) -> None:
        zero_based = getattr(trainer, "epoch", None)
        if zero_based == human_epoch - 1:
            raise ControlledTrainingPause(f"controlled pause after epoch {human_epoch}")

    return pause_after_save


def _load_config_mapping(source: Path) -> Mapping[object, object]:
    try:
        content = source.read_text(encoding="utf-8")
        node = yaml.compose(content, Loader=yaml.SafeLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise ResumableTrainingError(f"could not load resumable config: {source}") from exc
    if not isinstance(node, MappingNode):
        raise ResumableTrainingError("resumable config root must be a mapping")

    declared_keys: set[str] = set()
    for key_node, _ in node.value:
        if not isinstance(key_node, ScalarNode):
            raise ResumableTrainingError("resumable config keys must be strings")
        if key_node.value == "<<" or key_node.tag == "tag:yaml.org,2002:merge":
            raise ResumableTrainingError("resumable config must not use YAML merge keys")
        if key_node.value in declared_keys:
            raise ResumableTrainingError(f"resumable config has duplicate key: {key_node.value}")
        declared_keys.add(key_node.value)

    try:
        value = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise ResumableTrainingError(f"could not load resumable config: {source}") from exc
    if not isinstance(value, Mapping):
        raise ResumableTrainingError("resumable config root must be a mapping")
    return value


@dataclass(frozen=True)
class ResumableTrainingConfig:
    experiment_id: str
    experiment_group: str
    model: str
    data: Path
    prediction_source: Path
    output_root: Path
    epochs: int
    pause_after_epoch: int | None
    imgsz: int
    rect: bool
    batch: int
    seed: int
    workers: int
    device: int | str
    amp: bool
    cache: bool

    @classmethod
    def from_yaml(cls, path: Path) -> ResumableTrainingConfig:
        source = Path(path)
        value = _load_config_mapping(source)

        keys = set(value)
        missing = sorted(_CONFIG_KEYS - keys)
        unknown = sorted(str(key) for key in keys - _CONFIG_KEYS)
        if missing:
            raise ResumableTrainingError(
                f"resumable config is missing required keys: {', '.join(missing)}"
            )
        if unknown:
            raise ResumableTrainingError(
                f"resumable config has unknown keys: {', '.join(unknown)}"
            )

        for key in _STRING_KEYS:
            _require_non_empty_string(value[key], label=key)
        for key in _BOOLEAN_KEYS:
            if type(value[key]) is not bool:
                raise ResumableTrainingError(f"{key} must be a boolean")
        for key in _INTEGER_KEYS:
            _require_integer(value[key], label=key)
        pause_after_epoch = value["pause_after_epoch"]
        if pause_after_epoch is not None:
            _require_integer(pause_after_epoch, label="pause_after_epoch")
        device = value["device"]
        if not (type(device) is int or (type(device) is str and device.strip())):
            raise ResumableTrainingError("device must be an integer or non-empty string")

        repository = _repository_root()
        config = cls(
            experiment_id=value["experiment_id"],
            experiment_group=value["experiment_group"],
            model=value["model"],
            data=_resolve_config_path(value["data"], repository=repository),
            prediction_source=_resolve_config_path(
                value["prediction_source"], repository=repository
            ),
            output_root=_resolve_config_path(value["output_root"], repository=repository),
            epochs=value["epochs"],
            pause_after_epoch=pause_after_epoch,
            imgsz=value["imgsz"],
            rect=value["rect"],
            batch=value["batch"],
            seed=value["seed"],
            workers=value["workers"],
            device=device,
            amp=value["amp"],
            cache=value["cache"],
        )
        config._validate()
        return config

    def _validate(self) -> None:
        for key in ("experiment_id", "experiment_group", "model"):
            _require_non_empty_string(getattr(self, key), label=key)
        for key in ("data", "prediction_source", "output_root"):
            if not isinstance(getattr(self, key), Path):
                raise ResumableTrainingError(f"{key} must be a path")
        for key in _INTEGER_KEYS:
            _require_integer(getattr(self, key), label=key)
        for key in _BOOLEAN_KEYS:
            if type(getattr(self, key)) is not bool:
                raise ResumableTrainingError(f"{key} must be a boolean")
        if self.pause_after_epoch is not None:
            _require_integer(self.pause_after_epoch, label="pause_after_epoch")
        if not (type(self.device) is int or (type(self.device) is str and self.device.strip())):
            raise ResumableTrainingError("device must be an integer or non-empty string")
        if self.epochs < 2:
            raise ResumableTrainingError("epochs must be at least two")
        if self.imgsz <= 0:
            raise ResumableTrainingError("imgsz must be positive")
        if self.batch <= 0:
            raise ResumableTrainingError("batch must be positive")
        if self.workers < 0:
            raise ResumableTrainingError("workers must be greater than or equal to zero")
        if self.pause_after_epoch is not None and not 1 <= self.pause_after_epoch < self.epochs:
            raise ResumableTrainingError("pause_after_epoch must be before the final epoch")
        if _has_test_component(self.data) or _has_test_component(self.prediction_source):
            raise ResumableTrainingError("data and prediction_source must not include a test path")
        preflight_root = (_repository_root() / "runs" / "preflight").resolve(strict=False)
        if not _is_strict_child(self.output_root.resolve(strict=False), preflight_root):
            raise ResumableTrainingError("output_root must be a strict child of runs/preflight")
        for key in ("data", "prediction_source", "output_root"):
            _repository_relative(getattr(self, key), label=key)

    def to_dict(self) -> dict[str, object]:
        self._validate()
        return {
            "experiment_id": self.experiment_id,
            "experiment_group": self.experiment_group,
            "model": self.model,
            "data": _repository_relative(self.data, label="data"),
            "prediction_source": _repository_relative(
                self.prediction_source, label="prediction_source"
            ),
            "output_root": _repository_relative(self.output_root, label="output_root"),
            "epochs": self.epochs,
            "pause_after_epoch": self.pause_after_epoch,
            "imgsz": self.imgsz,
            "rect": self.rect,
            "batch": self.batch,
            "seed": self.seed,
            "workers": self.workers,
            "device": self.device,
            "amp": self.amp,
            "cache": self.cache,
        }

    def train_kwargs(self) -> dict[str, object]:
        self._validate()
        return {
            "epochs": self.epochs,
            "data": str(self.data),
            "imgsz": self.imgsz,
            "rect": self.rect,
            "batch": self.batch,
            "seed": self.seed,
            "workers": self.workers,
            "device": self.device,
            "amp": self.amp,
            "cache": self.cache,
            "project": str(self.output_root),
            "name": "train",
            "exist_ok": False,
            "save": True,
            "val": True,
        }


@dataclass(frozen=True)
class RunState:
    experiment_id: str
    status: Literal["running", "paused", "completed", "failed"]
    phase: Literal["initial", "resume", "finalize"]
    target_epochs: int
    last_completed_epoch: int
    checkpoint: str | None
    config_sha256: str
    dataset_provenance_sha256: str
    error: str | None

    def to_dict(self) -> dict[str, object]:
        _require_non_empty_string(self.experiment_id, label="experiment_id")
        if self.status not in _STATUSES:
            raise ResumableTrainingError("run state status is invalid")
        if self.phase not in _PHASES:
            raise ResumableTrainingError("run state phase is invalid")
        _require_integer(self.target_epochs, label="target_epochs")
        _require_integer(self.last_completed_epoch, label="last_completed_epoch")
        if self.target_epochs < 2:
            raise ResumableTrainingError("target_epochs must be at least two")
        if not 0 <= self.last_completed_epoch <= self.target_epochs:
            raise ResumableTrainingError(
                "last_completed_epoch must be between zero and target_epochs"
            )
        if self.checkpoint is not None:
            _require_non_empty_string(self.checkpoint, label="checkpoint")
            checkpoint = _resolve_config_path(self.checkpoint, repository=_repository_root())
            preflight_root = (_repository_root() / "runs" / "preflight").resolve(strict=False)
            if not _is_strict_child(checkpoint, preflight_root):
                raise ResumableTrainingError("checkpoint must be a strict child of runs/preflight")
            checkpoint_value: str | None = _repository_relative(checkpoint, label="checkpoint")
        else:
            checkpoint_value = None
        _require_sha256(self.config_sha256, label="config_sha256")
        _require_sha256(self.dataset_provenance_sha256, label="dataset_provenance_sha256")
        if self.error is not None and type(self.error) is not str:
            raise ResumableTrainingError("error must be a string or null")
        return {
            "experiment_id": self.experiment_id,
            "status": self.status,
            "phase": self.phase,
            "target_epochs": self.target_epochs,
            "last_completed_epoch": self.last_completed_epoch,
            "checkpoint": checkpoint_value,
            "config_sha256": self.config_sha256,
            "dataset_provenance_sha256": self.dataset_provenance_sha256,
            "error": self.error,
        }


def config_sha256(config: ResumableTrainingConfig) -> str:
    """Return the stable digest of the repository-relative configuration."""
    encoded = json.dumps(
        config.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _existing_state(path: Path) -> Mapping[str, object] | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise ResumableTrainingError("existing run state must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResumableTrainingError("existing run state is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ResumableTrainingError("existing run state must be a JSON object")
    return value


def _validate_transition(existing: Mapping[str, object], state: RunState) -> None:
    old_status = existing.get("status")
    old_phase = existing.get("phase")
    if old_status not in _STATUSES or old_phase not in _PHASES:
        raise ResumableTrainingError("existing run state has invalid status or phase")
    if old_status == "completed" and state.status != "completed":
        raise ResumableTrainingError("completed run state cannot transition to another status")
    if old_status == "paused" and state.phase == "initial":
        raise ResumableTrainingError("paused run state cannot transition back to initial")


def write_run_state(path: Path, state: RunState) -> None:
    """Atomically publish state after refusing forbidden state regressions."""
    destination = Path(path)
    payload = state.to_dict()
    existing = _existing_state(destination)
    if existing is not None:
        _validate_transition(existing, state)
    write_json_atomic(destination, payload)


def _default_model_factory(weights: str) -> object:
    from ultralytics import YOLO

    return YOLO(weights)


def _expected_prediction_count(evidence: Mapping[str, object]) -> int:
    content = evidence.get("content")
    if not isinstance(content, Mapping):
        raise ResumableTrainingError("dataset provenance content must be a mapping")
    counts = content.get("counts")
    if not isinstance(counts, Mapping):
        raise ResumableTrainingError("dataset provenance must declare split counts")
    count = counts.get("val")
    if type(count) is not int or count <= 0:
        raise ResumableTrainingError("dataset provenance val count must be a positive integer")
    return count


def _validate_weight(path: Path, *, output_root: Path, label: str) -> Path:
    try:
        validated = _validated_checkpoint_path(path, allowed_root=output_root)
        if validated.stat().st_size <= 0:
            raise ResumableTrainingError(f"{label} must be non-empty")
    except ResumableTrainingError as exc:
        raise ResumableTrainingError(
            f"{label} must be a non-empty regular non-symlink file"
        ) from exc
    return validated


def _trainer_argument(trainer: object, name: str) -> object:
    arguments = getattr(trainer, "args", None)
    if isinstance(arguments, Mapping):
        return arguments.get(name)
    return getattr(arguments, name, None)


def _validate_final_trainer(
    trainer: object,
    config: ResumableTrainingConfig,
    *,
    pause_checkpoint: Path,
) -> Path:
    if trainer is None:
        raise ResumableTrainingError("resumed model did not expose its trainer")
    expected = {"data": str(config.data), "seed": config.seed, "epochs": config.epochs}
    for name, value in expected.items():
        if _trainer_argument(trainer, name) != value:
            raise ResumableTrainingError(f"final trainer argument {name} changed")
    model_value = _trainer_argument(trainer, "model")
    model_is_resume_checkpoint = False
    if isinstance(model_value, str):
        model_is_resume_checkpoint = Path(model_value).resolve(strict=False) == (
            pause_checkpoint.resolve(strict=False)
        )
    if model_value != config.model and not model_is_resume_checkpoint:
        raise ResumableTrainingError("final trainer argument model family changed")
    save_dir_value = getattr(trainer, "save_dir", None)
    try:
        return Path(save_dir_value)
    except TypeError as exc:
        raise ResumableTrainingError("resumed trainer save_dir is invalid") from exc


def _dataset_identity(evidence: Mapping[str, object]) -> dict[str, str]:
    selection_hashes = evidence.get("selection_sha256")
    if not isinstance(selection_hashes, Mapping):
        raise ResumableTrainingError("dataset selection hashes are missing")
    values = {
        "dataset_provenance_sha256": evidence.get("sha256"),
        "data_yaml_sha256": evidence.get("data_yaml_sha256"),
        "selected_train_sha256": selection_hashes.get("train"),
        "selected_val_sha256": selection_hashes.get("val"),
    }
    identity: dict[str, str] = {}
    for label, value in values.items():
        identity[label] = _require_sha256(value, label=label)
    return identity


def _verify_run_identity(
    config: ResumableTrainingConfig,
    *,
    expected_config_sha256: str,
    expected_dataset_identity: Mapping[str, str],
) -> Mapping[str, object]:
    if config_sha256(config) != expected_config_sha256:
        raise ResumableTrainingError("training config changed between phases")
    evidence = read_dataset_provenance(config.data)
    current_identity = _dataset_identity(evidence)
    for label, expected in expected_dataset_identity.items():
        if current_identity.get(label) != expected:
            raise ResumableTrainingError(f"dataset {label} changed between phases")
    return evidence


def _prediction_result_is_nonempty(result: object) -> bool:
    if result is None:
        return False
    if isinstance(result, str | bytes | bytearray | Sequence | Mapping):
        return len(result) > 0
    return True


def _validate_prediction_evidence(
    predictions: object,
    *,
    prediction_directory: Path,
    expected_count: int,
) -> None:
    if (
        not isinstance(predictions, Sequence)
        or isinstance(predictions, str | bytes | bytearray)
        or len(predictions) != expected_count
        or any(not _prediction_result_is_nonempty(result) for result in predictions)
    ):
        raise ResumableTrainingError(
            f"prediction results must contain exactly {expected_count} nonempty objects"
        )
    if prediction_directory.is_symlink() or not prediction_directory.is_dir():
        raise ResumableTrainingError("saved prediction directory is missing or a symlink")
    extensions = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
    images = [
        path
        for path in prediction_directory.rglob("*")
        if path.suffix.lower() in extensions
    ]
    if len(images) != expected_count:
        raise ResumableTrainingError(
            f"saved prediction images must contain exactly {expected_count} files"
        )
    for image in images:
        if _has_symlink_component(image) or not image.is_file() or image.stat().st_size <= 0:
            raise ResumableTrainingError("saved prediction images must be nonempty regular files")


def _failure_state(
    config: ResumableTrainingConfig,
    *,
    phase: Literal["initial", "resume", "finalize"],
    last_completed_epoch: int,
    checkpoint: Path | None,
    config_digest: str,
    provenance_digest: str,
    error: str,
) -> RunState:
    return RunState(
        experiment_id=config.experiment_id,
        status="failed",
        phase=phase,
        target_epochs=config.epochs,
        last_completed_epoch=last_completed_epoch,
        checkpoint=None if checkpoint is None else str(checkpoint.resolve()),
        config_sha256=config_digest,
        dataset_provenance_sha256=provenance_digest,
        error=error,
    )


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _require_single_link(file_stat: os.stat_result, *, label: str) -> None:
    if file_stat.st_nlink != 1:
        raise ResumableTrainingError(f"{label} must have exactly one hard link")


@dataclass(frozen=True)
class _JsonPublicationToken:
    path: Path
    owned_stat: os.stat_result
    expected_bytes: bytes
    label: str


class _JsonPublicationError(ResumableTrainingError):
    def __init__(
        self,
        message: str,
        *,
        stage: Literal[
            "pre-replace", "post-replace-owned", "post-replace-ownership-lost"
        ],
        token: _JsonPublicationToken | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.token = token


class _EvidenceTransactionIntegrityError(ResumableTrainingError):
    def __init__(self, message: str, *, record_failed_state: bool) -> None:
        super().__init__(message)
        self.record_failed_state = record_failed_state


def _verify_owned_json_publication(token: _JsonPublicationToken) -> None:
    path = token.path
    label = token.label
    if _has_symlink_component(path):
        raise ResumableTrainingError(f"published {label} path must not contain symlinks")
    try:
        with path.open("rb") as handle:
            opened_stat = os.fstat(handle.fileno())
            _require_single_link(opened_stat, label=f"published {label}")
            if not os.path.samestat(opened_stat, token.owned_stat):
                raise ResumableTrainingError(f"published {label} identity changed")
            _require_path_matches_open_file(
                path,
                opened_stat,
                label=f"published {label}",
            )
            if handle.read() != token.expected_bytes:
                raise ResumableTrainingError(f"published {label} bytes changed")
            ending_stat = os.fstat(handle.fileno())
            _require_single_link(ending_stat, label=f"published {label}")
            if not os.path.samestat(ending_stat, token.owned_stat):
                raise ResumableTrainingError(f"published {label} identity changed")
            _require_path_matches_open_file(
                path,
                ending_stat,
                label=f"published {label}",
            )
    except OSError as exc:
        raise ResumableTrainingError(
            f"could not verify published {label}: {type(exc).__name__}: {exc}"
        ) from exc


def _verify_strict_json_bytes_at_path(
    path: Path, expected_bytes: bytes, *, label: str
) -> None:
    if _has_symlink_component(path):
        raise ResumableTrainingError(f"{label} path must not contain symlinks")
    try:
        with path.open("rb") as handle:
            opened_stat = os.fstat(handle.fileno())
            _require_single_link(opened_stat, label=label)
            _require_path_matches_open_file(path, opened_stat, label=label)
            if handle.read() != expected_bytes:
                raise ResumableTrainingError(f"{label} bytes changed")
            ending_stat = os.fstat(handle.fileno())
            if not os.path.samestat(opened_stat, ending_stat):
                raise ResumableTrainingError(f"{label} identity changed")
            _require_single_link(ending_stat, label=label)
            _require_path_matches_open_file(path, ending_stat, label=label)
    except OSError as exc:
        raise ResumableTrainingError(
            f"could not verify {label}: {type(exc).__name__}: {exc}"
        ) from exc


def _publication_path_is_owned(token: _JsonPublicationToken) -> bool:
    if _has_symlink_component(token.path):
        return False
    try:
        current_stat = token.path.lstat()
    except OSError:
        return False
    return stat.S_ISREG(current_stat.st_mode) and os.path.samestat(
        current_stat, token.owned_stat
    )


def _rollback_owned_json_publication(token: _JsonPublicationToken) -> bool:
    if not _publication_path_is_owned(token):
        return False
    try:
        _unlink_path_if_owned(token.path, token.owned_stat)
    except OSError:
        return False
    try:
        token.path.lstat()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return False


def _verification_publication_error(
    token: _JsonPublicationToken, error: Exception
) -> _JsonPublicationError:
    if _publication_path_is_owned(token):
        stage: Literal["post-replace-owned", "post-replace-ownership-lost"] = (
            "post-replace-owned"
        )
    else:
        stage = "post-replace-ownership-lost"
    return _JsonPublicationError(
        f"{token.label} publication {stage} verification failed: "
        f"{type(error).__name__}: {error}",
        stage=stage,
        token=token,
    )


def _reverify_owned_json_publication(token: _JsonPublicationToken) -> None:
    try:
        _verify_owned_json_publication(token)
    except Exception as exc:
        raise _verification_publication_error(token, exc) from exc


def _publish_owned_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    label: str,
    previous_token: _JsonPublicationToken | None = None,
) -> _JsonPublicationToken:
    destination = Path(path)
    if _has_symlink_component(destination):
        raise _JsonPublicationError(
            f"{label} path must not contain symlinks", stage="pre-replace"
        )

    expected_bytes = _canonical_json_bytes(payload)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    if _has_symlink_component(temporary):
        raise _JsonPublicationError(
            f"{label} temporary path must not contain symlinks", stage="pre-replace"
        )
    if temporary.exists() or temporary.is_symlink():
        raise _JsonPublicationError(
            f"{label} temporary path must not already exist", stage="pre-replace"
        )

    temporary_created = False
    owned_stat: os.stat_result | None = None
    replace_completed = False
    publication_verified = False
    primary_failed = False
    try:
        try:
            with temporary.open("x+b") as handle:
                temporary_created = True
                initial_stat = os.fstat(handle.fileno())
                _require_single_link(initial_stat, label=f"{label} temporary file")
                handle.write(expected_bytes)
                handle.flush()
                os.fsync(handle.fileno())
                handle.seek(0)
                if handle.read() != expected_bytes:
                    raise ResumableTrainingError(f"{label} temporary bytes changed")
                owned_stat = os.fstat(handle.fileno())
                _require_single_link(owned_stat, label=f"{label} temporary file")
                if not os.path.samestat(initial_stat, owned_stat):
                    raise ResumableTrainingError(f"{label} temporary identity changed")
                _require_path_matches_open_file(
                    temporary,
                    owned_stat,
                    label=f"{label} temporary file",
                )
        except FileExistsError as exc:
            raise _JsonPublicationError(
                f"{label} temporary path must not already exist",
                stage="pre-replace",
            ) from exc
        except OSError as exc:
            raise _JsonPublicationError(
                f"could not write {label} temporary file: {type(exc).__name__}: {exc}",
                stage="pre-replace",
            ) from exc

        if owned_stat is None:
            raise ResumableTrainingError(f"{label} temporary identity is missing")
        if _has_symlink_component(temporary) or _has_symlink_component(destination):
            raise ResumableTrainingError(f"{label} publication path contains a symlink")
        current_temporary_stat = _require_path_matches_open_file(
            temporary,
            owned_stat,
            label=f"{label} temporary file",
        )
        _require_single_link(current_temporary_stat, label=f"{label} temporary file")
        try:
            if previous_token is None:
                os.link(temporary, destination, follow_symlinks=False)
            else:
                _replace_owned_json_path(temporary, destination, previous_token)
        except OSError as exc:
            raise _JsonPublicationError(
                f"{label} publication pre-replace failed: {type(exc).__name__}: {exc}",
                stage="pre-replace",
            ) from exc
        replace_completed = True
        token = _JsonPublicationToken(
            path=destination,
            owned_stat=owned_stat,
            expected_bytes=expected_bytes,
            label=label,
        )
        if previous_token is None:
            _unlink_path_if_owned(temporary, owned_stat)
        _reverify_owned_json_publication(token)
        publication_verified = True
        return token
    except _JsonPublicationError:
        primary_failed = True
        raise
    except Exception as exc:
        primary_failed = True
        if replace_completed and owned_stat is not None:
            token = _JsonPublicationToken(
                path=destination,
                owned_stat=owned_stat,
                expected_bytes=expected_bytes,
                label=label,
            )
            raise _verification_publication_error(token, exc) from exc
        raise _JsonPublicationError(
            f"{label} publication pre-replace failed: {type(exc).__name__}: {exc}",
            stage="pre-replace",
        ) from exc
    except BaseException:
        primary_failed = True
        raise
    finally:
        if temporary_created and owned_stat is not None:
            try:
                _unlink_path_if_owned(temporary, owned_stat)
            except Exception:
                if not primary_failed and not (replace_completed and publication_verified):
                    raise


def _publish_completion_evidence(
    path: Path, payload: Mapping[str, object]
) -> _JsonPublicationToken:
    return _publish_owned_json(
        path,
        payload,
        label="completion evidence",
    )


def _publish_run_state_transition(
    path: Path,
    state: RunState,
    *,
    previous_token: _JsonPublicationToken | None,
    label: str = "run state",
) -> _JsonPublicationToken:
    destination = Path(path)
    payload = state.to_dict()
    if previous_token is None:
        if destination.exists() or destination.is_symlink():
            raise _JsonPublicationError(
                "run state publication pre-replace refused an unowned destination",
                stage="pre-replace",
            )
    else:
        if previous_token.path != destination:
            raise ResumableTrainingError("previous run state token path changed")
        try:
            existing = json.loads(previous_token.expected_bytes)
        except json.JSONDecodeError as exc:
            raise ResumableTrainingError("previous run state token is invalid") from exc
        if not isinstance(existing, Mapping):
            raise ResumableTrainingError("previous run state token must contain an object")
        _validate_transition(existing, state)
        _reverify_owned_json_publication(previous_token)
    return _publish_owned_json(
        destination,
        payload,
        label=label,
        previous_token=previous_token,
    )


def _publish_completed_run_state(
    path: Path,
    state: RunState,
    *,
    previous_token: _JsonPublicationToken,
) -> _JsonPublicationToken:
    token = _publish_run_state_transition(
        path,
        state,
        previous_token=previous_token,
        label="completed state",
    )
    return token


def _recover_training_evidence_transaction(
    error: Exception,
    *,
    completion_token: _JsonPublicationToken | None,
    state_token: _JsonPublicationToken | None,
) -> None:
    failed_token = error.token if isinstance(error, _JsonPublicationError) else None
    if failed_token is not None and failed_token.label == "completed state":
        state_token = failed_token
    if failed_token is not None and failed_token.label == "completion evidence":
        completion_token = failed_token

    if state_token is not None:
        state_owned = _publication_path_is_owned(state_token)
        completion_exists = completion_token is not None and (
            completion_token.path.exists() or completion_token.path.is_symlink()
        )
        completion_owned = completion_token is not None and _publication_path_is_owned(
            completion_token
        )
        if not state_owned and (
            state_token.path.exists() or state_token.path.is_symlink()
        ):
            raise _EvidenceTransactionIntegrityError(
                "training evidence integrity error: completed state ownership was lost "
                "after publication; retained completion evidence and did not attempt a "
                f"completed-to-failed transition; original error: {error}",
                record_failed_state=False,
            ) from error
        if state_owned and completion_exists and not completion_owned:
            try:
                _verify_strict_json_bytes_at_path(
                    completion_token.path,
                    completion_token.expected_bytes,
                    label="foreign completion evidence",
                )
            except Exception as completion_error:
                if not _rollback_owned_json_publication(state_token):
                    raise _EvidenceTransactionIntegrityError(
                        "training evidence integrity error: invalid foreign completion "
                        "was retained and completed state could not be safely rolled back; "
                        f"original errors: {error}; {completion_error}",
                        record_failed_state=False,
                    ) from error
                raise _EvidenceTransactionIntegrityError(
                    "training evidence integrity error: invalid foreign completion was "
                    "retained after owned completed state rollback; a failed state must "
                    f"record the untrusted path; original errors: {error}; {completion_error}",
                    record_failed_state=True,
                ) from error
            else:
                raise _EvidenceTransactionIntegrityError(
                    "training evidence integrity error: completion evidence was replaced "
                    "after publication; retained the strict foreign completion path and "
                    f"completed state as an auditable evidence pair; original error: {error}",
                    record_failed_state=False,
                ) from error
        if state_owned and not _rollback_owned_json_publication(state_token):
            raise _EvidenceTransactionIntegrityError(
                "training evidence integrity error: completed state could not be "
                "safely rolled back; retained completion evidence and did not attempt "
                f"a completed-to-failed transition; original error: {error}",
                record_failed_state=False,
            ) from error

    if (
        state_token is not None
        and not _publication_path_is_owned(state_token)
        and (state_token.path.exists() or state_token.path.is_symlink())
    ):
        raise _EvidenceTransactionIntegrityError(
            "training evidence integrity error: completed state ownership was lost "
            "during recovery; completion evidence was retained",
            record_failed_state=False,
        ) from error

    if completion_token is not None and not _rollback_owned_json_publication(completion_token):
        raise _EvidenceTransactionIntegrityError(
            "training evidence integrity error: completion evidence ownership was lost "
            f"after publication; the foreign path was retained; original error: {error}",
            record_failed_state=True,
        ) from error


def _finalize_training_evidence(
    *,
    completion_path: Path,
    completion_payload: Mapping[str, object],
    state_path: Path,
    completed_state: RunState,
    previous_state_token: _JsonPublicationToken,
) -> None:
    completion_token: _JsonPublicationToken | None = None
    state_token: _JsonPublicationToken | None = None
    try:
        completion_token = _publish_completion_evidence(
            completion_path, completion_payload
        )
        state_token = _publish_completed_run_state(
            state_path,
            completed_state,
            previous_token=previous_state_token,
        )
        _reverify_owned_json_publication(completion_token)
        _reverify_owned_json_publication(state_token)
    except Exception as exc:
        failed_token = exc.token if isinstance(exc, _JsonPublicationError) else None
        if (
            state_token is None
            and failed_token is not None
            and failed_token.label == "completion evidence"
            and isinstance(exc, _JsonPublicationError)
            and exc.stage == "post-replace-ownership-lost"
        ):
            try:
                _verify_strict_json_bytes_at_path(
                    failed_token.path,
                    failed_token.expected_bytes,
                    label="foreign completion evidence",
                )
                state_token = _publish_completed_run_state(
                    state_path,
                    completed_state,
                    previous_token=previous_state_token,
                )
                _verify_strict_json_bytes_at_path(
                    failed_token.path,
                    failed_token.expected_bytes,
                    label="foreign completion evidence",
                )
                _reverify_owned_json_publication(state_token)
            except Exception as recovery_error:
                _recover_training_evidence_transaction(
                    recovery_error,
                    completion_token=failed_token,
                    state_token=state_token,
                )
                raise
            raise _EvidenceTransactionIntegrityError(
                "training evidence integrity error: completion ownership was lost "
                "before state publication; retained the strict foreign completion "
                "and owned completed state as an auditable evidence pair",
                record_failed_state=False,
            ) from exc
        _recover_training_evidence_transaction(
            exc,
            completion_token=completion_token,
            state_token=state_token,
        )
        raise


def run_resumable_training(
    config: ResumableTrainingConfig,
    *,
    model_factory: Callable[[str], object] | None = None,
    checkpoint_loader: Callable[[Path], object] | None = None,
) -> TrainingRunResult:
    """Run the fixed five-epoch pause/resume preflight and publish completion evidence."""
    config._validate()
    if config.epochs != 5 or config.pause_after_epoch != 2:
        raise ResumableTrainingError(
            "resumable preflight requires five epochs and a pause after epoch two"
        )
    output_root = config.output_root
    completion_path = output_root / "completion.json"
    state_path = output_root / "run_state.json"
    if output_root.exists() or output_root.is_symlink():
        raise ResumableTrainingError("pre-existing output root is not allowed")
    output_root.mkdir(parents=True, exist_ok=False)
    lock_handle = (output_root / ".training-evidence.lock").open("x+b")
    lock_handle.write(b"rgbt-uav resumable training evidence single-writer lock\n")
    lock_handle.flush()
    os.fsync(lock_handle.fileno())

    factory = _default_model_factory if model_factory is None else model_factory
    tracker = _TimingTracker()
    started_at = time.perf_counter()
    phase: Literal["initial", "resume", "finalize"] = "initial"
    pause_checkpoint: Path | None = None
    config_digest = config_sha256(config)
    provenance_digest = "0" * 64
    dataset_identity: dict[str, str] = {}
    run_state_token: _JsonPublicationToken | None = None
    try:
        evidence = read_dataset_provenance(config.data)
        dataset_identity = _dataset_identity(evidence)
        provenance_digest = dataset_identity["dataset_provenance_sha256"]
        expected_predictions = _expected_prediction_count(evidence)
        try:
            prediction_source = config.prediction_source.resolve(strict=True)
        except OSError as exc:
            raise ResumableTrainingError("prediction source directory does not exist") from exc
        if prediction_source != evidence.get("val_directory"):
            raise ResumableTrainingError(
                "prediction source must exactly match the data YAML val directory"
            )

        run_state_token = _publish_run_state_transition(
            state_path,
            RunState(
                experiment_id=config.experiment_id,
                status="running",
                phase="initial",
                target_epochs=config.epochs,
                last_completed_epoch=0,
                checkpoint=None,
                config_sha256=config_digest,
                dataset_provenance_sha256=provenance_digest,
                error=None,
            ),
            previous_token=None,
        )
        initial = factory(config.model)
        _attach_timing_callbacks(initial, tracker, "initial")
        add_callback = getattr(initial, "add_callback", None)
        if not callable(add_callback):
            raise ResumableTrainingError("training model does not support callbacks")
        add_callback("on_model_save", make_pause_callback(human_epoch=2))
        last_path = output_root / "train" / "weights" / "last.pt"
        try:
            initial.train(**config.train_kwargs())
        except ControlledTrainingPause:
            inspect_resumable_checkpoint(
                last_path,
                allowed_root=output_root,
                expected_target_epochs=config.epochs,
                expected_human_epoch=2,
                loader=checkpoint_loader,
            )
            checkpoint_directory = output_root / "checkpoints"
            checkpoint_directory.mkdir(parents=True, exist_ok=False)
            pause_checkpoint = preserve_resume_checkpoint(
                last_path, checkpoint_directory / "epoch2_resume.pt"
            )
            inspect_resumable_checkpoint(
                pause_checkpoint,
                allowed_root=output_root,
                expected_target_epochs=config.epochs,
                expected_human_epoch=2,
                loader=checkpoint_loader,
            )
            if tracker.completed_epochs() != [1, 2]:
                raise ResumableTrainingError(
                    "initial epoch sequence must be exactly [1, 2]"
                ) from None
            run_state_token = _publish_run_state_transition(
                state_path,
                RunState(
                    experiment_id=config.experiment_id,
                    status="paused",
                    phase="initial",
                    target_epochs=config.epochs,
                    last_completed_epoch=2,
                    checkpoint=str(pause_checkpoint.resolve()),
                    config_sha256=config_digest,
                    dataset_provenance_sha256=provenance_digest,
                    error=None,
                ),
                previous_token=run_state_token,
            )
        else:
            raise ResumableTrainingError("initial phase reached return without controlled pause")

        phase = "resume"
        resumed = factory(str(pause_checkpoint))
        _verify_run_identity(
            config,
            expected_config_sha256=config_digest,
            expected_dataset_identity=dataset_identity,
        )
        run_state_token = _publish_run_state_transition(
            state_path,
            RunState(
                experiment_id=config.experiment_id,
                status="running",
                phase="resume",
                target_epochs=config.epochs,
                last_completed_epoch=2,
                checkpoint=str(pause_checkpoint.resolve()),
                config_sha256=config_digest,
                dataset_provenance_sha256=provenance_digest,
                error=None,
            ),
            previous_token=run_state_token,
        )
        _attach_timing_callbacks(resumed, tracker, "resume")
        resumed.train(
            resume=str(pause_checkpoint),
            imgsz=config.imgsz,
            batch=config.batch,
            device=config.device,
            close_mosaic=10,
        )
        epoch_timings = tracker.epoch_timings()
        completed_epochs = [timing.human_epoch for timing in epoch_timings]
        if completed_epochs != [1, 2, 3, 4, 5]:
            raise ResumableTrainingError(
                "recorded epoch sequence must be exactly [1, 2, 3, 4, 5]"
            )
        if tracker.framework_final_validation_seconds is None:
            raise ResumableTrainingError("framework final validation timing is missing")
        save_dir = _validate_final_trainer(
            getattr(resumed, "trainer", None),
            config,
            pause_checkpoint=pause_checkpoint,
        )
        _verify_run_identity(
            config,
            expected_config_sha256=config_digest,
            expected_dataset_identity=dataset_identity,
        )

        phase = "finalize"
        best_weights = _validate_weight(
            save_dir / "weights" / "best.pt", output_root=output_root, label="best.pt"
        )
        last_weights = _validate_weight(
            save_dir / "weights" / "last.pt", output_root=output_root, label="last.pt"
        )
        validation_started = time.perf_counter()
        validation = resumed.val(
            data=str(config.data),
            split="val",
            imgsz=config.imgsz,
            batch=config.batch,
            device=config.device,
            workers=0,
        )
        final_validation_seconds = time.perf_counter() - validation_started
        if not math.isfinite(final_validation_seconds) or final_validation_seconds < 0:
            raise ResumableTrainingError("final validation timing must be finite and non-negative")
        metrics = read_finite_metrics(getattr(validation, "results_dict", None))
        try:
            precision = metrics["metrics/precision(B)"]
            recall = metrics["metrics/recall(B)"]
        except KeyError as exc:
            raise ResumableTrainingError(
                "validation metrics require precision and recall"
            ) from exc
        metrics["f1"] = 0.0 if precision + recall == 0 else 2 * precision * recall / (
            precision + recall
        )
        if not math.isfinite(metrics["f1"]):
            raise ResumableTrainingError("derived f1 metric must be finite")

        reload_started = time.perf_counter()
        reloaded = factory(str(best_weights))
        prediction_directory = output_root / "reload_prediction"
        predictions = reloaded.predict(
            source=str(config.prediction_source),
            save=True,
            project=str(output_root),
            name="reload_prediction",
        )
        _validate_prediction_evidence(
            predictions,
            prediction_directory=prediction_directory,
            expected_count=expected_predictions,
        )
        reload_prediction_seconds = time.perf_counter() - reload_started
        if not math.isfinite(reload_prediction_seconds) or reload_prediction_seconds < 0:
            raise ResumableTrainingError(
                "reload prediction timing must be finite and non-negative"
            )
        final_evidence = _verify_run_identity(
            config,
            expected_config_sha256=config_digest,
            expected_dataset_identity=dataset_identity,
        )

        elapsed_seconds = time.perf_counter() - started_at
        train_seconds = [timing.train_seconds for timing in epoch_timings]
        val_seconds = [timing.val_seconds for timing in epoch_timings]
        measured_seconds = (
            sum(train_seconds)
            + sum(val_seconds)
            + tracker.framework_final_validation_seconds
            + final_validation_seconds
            + reload_prediction_seconds
        )
        if measured_seconds > elapsed_seconds + 0.001:
            raise ResumableTrainingError("measured timing exceeds total elapsed time")
        fixed_seconds = max(0.0, elapsed_seconds - measured_seconds)
        git_commit, git_error = git_provenance()
        timing = {
            "train_epoch_seconds": train_seconds,
            "val_epoch_seconds": val_seconds,
            "framework_final_validation_seconds": tracker.framework_final_validation_seconds,
            "final_validation_seconds": final_validation_seconds,
            "reload_prediction_seconds": reload_prediction_seconds,
            "fixed_seconds": fixed_seconds,
        }
        payload: dict[str, object] = {
            "status": "completed",
            "experiment_id": config.experiment_id,
            "experiment_group": config.experiment_group,
            "config": config.to_dict(),
            "completed_epochs": completed_epochs,
            "pause_checkpoint": str(pause_checkpoint.resolve()),
            "pause_checkpoint_sha256": _sha256_file(pause_checkpoint),
            "best_weights": str(best_weights.resolve()),
            "best_weights_sha256": _sha256_file(best_weights),
            "last_weights": str(last_weights.resolve()),
            "last_weights_sha256": _sha256_file(last_weights),
            "metrics": metrics,
            "elapsed_seconds": elapsed_seconds,
            "timing": timing,
            "peak_cuda_memory_bytes": peak_cuda_memory_bytes(),
            "git_commit": git_commit,
            "environment": environment_snapshot(),
            "dataset_provenance": final_evidence["content"],
            "dataset_provenance_path": final_evidence["path"],
            "dataset_provenance_sha256": final_evidence["sha256"],
            "data_yaml_sha256": final_evidence["data_yaml_sha256"],
            "selected_train_sha256": final_evidence["selection_sha256"]["train"],
            "selected_val_sha256": final_evidence["selection_sha256"]["val"],
            "selection_files": final_evidence["selection_files"],
        }
        if git_error is not None:
            payload["git_error"] = git_error
        completed_state = RunState(
            experiment_id=config.experiment_id,
            status="completed",
            phase="finalize",
            target_epochs=config.epochs,
            last_completed_epoch=5,
            checkpoint=str(pause_checkpoint.resolve()),
            config_sha256=config_digest,
            dataset_provenance_sha256=provenance_digest,
            error=None,
        )
        _finalize_training_evidence(
            completion_path=completion_path,
            completion_payload=payload,
            state_path=state_path,
            completed_state=completed_state,
            previous_state_token=run_state_token,
        )
        return TrainingRunResult(
            best_weights=best_weights,
            last_weights=last_weights,
            pause_checkpoint=pause_checkpoint,
            state_path=state_path,
            completion_path=completion_path,
            metrics=metrics,
            completed_epochs=completed_epochs,
            epoch_timings=epoch_timings,
        )
    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        if (
            isinstance(exc, _EvidenceTransactionIntegrityError)
            and not exc.record_failed_state
        ):
            raise ResumableTrainingError(error_text) from exc
        completed = set(tracker.completed_epochs())
        last_completed_epoch = 0
        for expected_epoch in range(1, config.epochs + 1):
            if expected_epoch not in completed:
                break
            last_completed_epoch = expected_epoch
        try:
            previous_failure_token: _JsonPublicationToken | None = None
            if run_state_token is not None:
                if _publication_path_is_owned(run_state_token):
                    previous_failure_token = run_state_token
                elif state_path.exists() or state_path.is_symlink():
                    raise _EvidenceTransactionIntegrityError(
                        "training evidence integrity error: current run state ownership "
                        "was lost; the foreign state path was retained",
                        record_failed_state=False,
                    )
            _publish_run_state_transition(
                state_path,
                _failure_state(
                    config,
                    phase=phase,
                    last_completed_epoch=last_completed_epoch,
                    checkpoint=pause_checkpoint,
                    config_digest=config_digest,
                    provenance_digest=provenance_digest,
                    error=error_text,
                ),
                previous_token=previous_failure_token,
            )
        except Exception as state_error:
            raise ResumableTrainingError(
                f"{error_text}; failed to publish failed state: "
                f"{type(state_error).__name__}: {state_error}"
            ) from exc
        raise ResumableTrainingError(error_text) from exc
    finally:
        lock_handle.close()
