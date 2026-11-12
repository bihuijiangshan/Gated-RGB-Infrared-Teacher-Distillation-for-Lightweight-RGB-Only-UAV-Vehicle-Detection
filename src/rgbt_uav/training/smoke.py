"""One-epoch Ultralytics training guarded by reproducibility checks."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path

import yaml

_CONFIG_KEYS = frozenset(
    {
        "experiment_group",
        "model",
        "data",
        "prediction_source",
        "output_root",
        "epochs",
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


def _resolve_config_path(declared: str, *, repository: Path) -> Path:
    path = Path(declared)
    if not path.is_absolute():
        path = repository / path
    return path.resolve(strict=False)


class SmokeTrainingError(RuntimeError):
    """Raised when a smoke training run fails a required completion gate."""


class SmokeTrainingCleanupError(SmokeTrainingError):
    """Preserves a run failure when completion-artifact cleanup also fails."""

    def __init__(self, original_error: Exception, cleanup_errors: list[Exception]) -> None:
        self.original_error = original_error
        self.cleanup_errors = tuple(cleanup_errors)
        super().__init__("smoke training failed and completion artifact cleanup also failed")


@dataclass(frozen=True)
class SmokeTrainingConfig:
    experiment_group: str
    model: str
    data: Path
    prediction_source: Path
    output_root: Path
    epochs: int
    imgsz: int
    rect: bool
    batch: int
    seed: int
    workers: int
    device: int | str
    amp: bool
    cache: bool
    _declared_data: str | None = field(default=None, repr=False, compare=False)
    _declared_prediction_source: str | None = field(default=None, repr=False, compare=False)
    _declared_output_root: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        for key in ("experiment_group", "model"):
            value = getattr(self, key)
            if type(value) is not str or not value.strip():
                raise SmokeTrainingError(f"{key} must be a non-empty string")
        for key in ("data", "prediction_source", "output_root"):
            value = getattr(self, key)
            if not isinstance(value, Path) or not str(value):
                raise SmokeTrainingError(f"{key} must be a non-empty path")
        for key in _INTEGER_KEYS:
            value = getattr(self, key)
            if type(value) is not int:
                raise SmokeTrainingError(f"{key} must be an integer")
        for key in _BOOLEAN_KEYS:
            if type(getattr(self, key)) is not bool:
                raise SmokeTrainingError(f"{key} must be a boolean")
        if self.epochs != 1:
            raise SmokeTrainingError("smoke config must use exactly one epoch")
        if self.imgsz <= 0:
            raise SmokeTrainingError("imgsz must be positive")
        if self.batch <= 0:
            raise SmokeTrainingError("batch must be positive")
        if self.workers < 0:
            raise SmokeTrainingError("workers must be greater than or equal to zero")
        if type(self.device) is int:
            return
        if type(self.device) is str and self.device.strip():
            return
        raise SmokeTrainingError("device must be an integer or non-empty string")

    @classmethod
    def from_yaml(cls, path: Path) -> SmokeTrainingConfig:
        source = Path(path)
        try:
            value = yaml.safe_load(source.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise SmokeTrainingError(f"could not load smoke config: {source}") from exc
        if not isinstance(value, Mapping):
            raise SmokeTrainingError("smoke config root must be a mapping")
        keys = set(value)
        missing = sorted(_CONFIG_KEYS - keys)
        unknown = sorted(str(key) for key in keys - _CONFIG_KEYS)
        if missing:
            raise SmokeTrainingError(f"smoke config is missing required keys: {', '.join(missing)}")
        if unknown:
            raise SmokeTrainingError(f"smoke config has unknown keys: {', '.join(unknown)}")
        for key in _BOOLEAN_KEYS:
            if type(value[key]) is not bool:
                raise SmokeTrainingError(f"{key} must be a boolean")
        for key in _INTEGER_KEYS:
            if type(value[key]) is not int:
                raise SmokeTrainingError(f"{key} must be an integer")
        for key in ("experiment_group", "model", "data", "prediction_source", "output_root"):
            if type(value[key]) is not str or not value[key].strip():
                raise SmokeTrainingError(f"{key} must be a non-empty string")
        device = value["device"]
        if not (type(device) is int or (type(device) is str and device.strip())):
            raise SmokeTrainingError("device must be an integer or non-empty string")
        repository = Path(__file__).resolve().parents[3]
        return cls(
            experiment_group=value["experiment_group"],
            model=value["model"],
            data=_resolve_config_path(value["data"], repository=repository),
            prediction_source=_resolve_config_path(
                value["prediction_source"], repository=repository
            ),
            output_root=_resolve_config_path(value["output_root"], repository=repository),
            epochs=value["epochs"],
            imgsz=value["imgsz"],
            rect=value["rect"],
            batch=value["batch"],
            seed=value["seed"],
            workers=value["workers"],
            device=device,
            amp=value["amp"],
            cache=value["cache"],
            _declared_data=value["data"],
            _declared_prediction_source=value["prediction_source"],
            _declared_output_root=value["output_root"],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_group": self.experiment_group,
            "model": self.model,
            "data": self._declared_data or self.data.as_posix(),
            "prediction_source": self._declared_prediction_source
            or self.prediction_source.as_posix(),
            "output_root": self._declared_output_root or self.output_root.as_posix(),
            "epochs": self.epochs,
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
        return {
            "data": str(self.data),
            "epochs": self.epochs,
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
        }


@dataclass(frozen=True)
class SmokeTrainingResult:
    best_weights: Path
    last_weights: Path
    completion_path: Path
    metrics: dict[str, float]


def _temporary_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".tmp")


def _unlink_file_if_present(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if not path.is_file() and not path.is_symlink():
        raise SmokeTrainingError(f"refusing to remove non-file completion artifact: {path}")
    path.unlink()


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    """Write deterministic JSON without altering an existing destination before replace."""
    destination = Path(path)
    temporary = _temporary_path(destination)
    try:
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"
        destination.parent.mkdir(parents=True, exist_ok=True)
        _unlink_file_if_present(temporary)
        temporary.write_text(serialized, encoding="utf-8", newline="\n")
        temporary.replace(destination)
    except Exception as write_error:
        cleanup_errors = _cleanup_atomic_temporary(temporary)
        if cleanup_errors:
            raise SmokeTrainingCleanupError(write_error, cleanup_errors) from write_error
        raise
    cleanup_errors = _cleanup_atomic_temporary(temporary)
    if cleanup_errors:
        published_error = SmokeTrainingError(
            "completion JSON was published but temporary cleanup left a residual artifact"
        )
        raise SmokeTrainingCleanupError(published_error, cleanup_errors) from published_error


def _cleanup_atomic_temporary(temporary: Path) -> list[Exception]:
    try:
        if temporary.exists() or temporary.is_symlink():
            _unlink_file_if_present(temporary)
    except Exception as exc:
        return [exc]
    return []


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_finite_json(value: object, *, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise SmokeTrainingError(f"{label} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if type(key) is not str:
                raise SmokeTrainingError(f"{label} must use string keys")
            _require_finite_json(nested, label=label)
    elif isinstance(value, list):
        for nested in value:
            _require_finite_json(nested, label=label)


def _read_dataset_provenance(data_path: Path) -> dict[str, object]:
    try:
        resolved_data_path = data_path.resolve(strict=True)
    except OSError as exc:
        raise SmokeTrainingError(f"data YAML does not exist: {data_path}") from exc
    try:
        data_value = yaml.safe_load(resolved_data_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SmokeTrainingError(f"could not parse data YAML: {resolved_data_path}") from exc
    if not isinstance(data_value, Mapping):
        raise SmokeTrainingError("data YAML must be a mapping")
    for key in ("path", "train", "val"):
        if type(data_value.get(key)) is not str or not data_value[key].strip():
            raise SmokeTrainingError(f"data YAML must contain a non-empty string {key}")
    modality_path = Path(data_value["path"])
    if not modality_path.is_absolute():
        modality_path = resolved_data_path.parent / modality_path
    try:
        modality_path = modality_path.resolve(strict=True)
    except OSError as exc:
        raise SmokeTrainingError(
            f"data YAML modality path does not exist: {modality_path}"
        ) from exc
    if not modality_path.is_dir():
        raise SmokeTrainingError(f"data YAML modality path is not a directory: {modality_path}")
    if modality_path.parent != resolved_data_path.parent:
        raise SmokeTrainingError(
            "data YAML modality path must be directly beneath the dataset root"
        )
    if modality_path.name != resolved_data_path.stem:
        raise SmokeTrainingError("data YAML modality directory must match the data YAML stem")
    _resolve_data_split_path(
        data_value["train"], data_root=modality_path, label="train"
    )
    val_directory = _resolve_data_split_path(
        data_value["val"], data_root=modality_path, label="val"
    )
    dataset_root = resolved_data_path.parent
    provenance_path = dataset_root / "provenance.json"
    if provenance_path.is_symlink() or not provenance_path.is_file():
        raise SmokeTrainingError(
            f"dataset provenance must be a regular non-symlink file: {provenance_path}"
        )
    try:
        resolved_provenance_path = provenance_path.resolve(strict=True)
    except OSError as exc:
        raise SmokeTrainingError(f"dataset provenance does not exist: {provenance_path}") from exc
    if resolved_provenance_path.parent != dataset_root:
        raise SmokeTrainingError("dataset provenance must be directly beneath the dataset root")
    try:
        provenance = json.loads(resolved_provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SmokeTrainingError(f"could not parse dataset provenance: {provenance_path}") from exc
    if not isinstance(provenance, Mapping):
        raise SmokeTrainingError("dataset provenance must be a JSON object")
    hashes = provenance.get("selection_sha256")
    if not isinstance(hashes, Mapping):
        raise SmokeTrainingError("dataset provenance must contain selection_sha256")
    for split in ("train", "val"):
        hash_value = hashes.get(split)
        if not _is_sha256(hash_value):
            raise SmokeTrainingError(f"dataset provenance is missing {split} selection hash")
    _require_finite_json(provenance, label="dataset provenance")
    selection_files = _selection_files(
        dataset_root,
        hashes=hashes,
        counts=provenance.get("counts"),
    )
    return {
        "path": str(resolved_provenance_path),
        "sha256": _sha256(resolved_provenance_path),
        "content": dict(provenance),
        "selection_sha256": {"train": hashes["train"], "val": hashes["val"]},
        "selection_files": selection_files,
        "data_yaml_sha256": _sha256(resolved_data_path),
        "modality_path": modality_path,
        "val_directory": val_directory,
    }


def _resolve_data_split_path(value: str, *, data_root: Path, label: str) -> Path:
    split_path = Path(value)
    if split_path.is_absolute():
        raise SmokeTrainingError(f"data YAML {label} split path must be relative")
    split_path = data_root / split_path
    try:
        resolved = split_path.resolve(strict=True)
    except OSError as exc:
        raise SmokeTrainingError(
            f"data YAML {label} image directory does not exist: {split_path}"
        ) from exc
    if not resolved.is_dir():
        raise SmokeTrainingError(f"data YAML {label} image path is not a directory: {resolved}")
    if not _is_strict_child(resolved, data_root):
        raise SmokeTrainingError(
            f"data YAML {label} split path must stay beneath the modality root"
        )
    return resolved


def _is_strict_child(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path != root


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _selection_files(
    dataset_root: Path,
    *,
    hashes: Mapping[object, object],
    counts: object,
) -> dict[str, dict[str, object]]:
    if counts is not None and not isinstance(counts, Mapping):
        raise SmokeTrainingError("dataset provenance counts must be a mapping")
    selection_directory = dataset_root / "selection"
    if selection_directory.is_symlink() or not selection_directory.is_dir():
        raise SmokeTrainingError(
            "selection directory must be a regular non-symlink directory: "
            f"{selection_directory}"
        )
    try:
        resolved_selection_directory = selection_directory.resolve(strict=True)
    except OSError as exc:
        raise SmokeTrainingError(
            f"selection directory does not exist: {selection_directory}"
        ) from exc
    if resolved_selection_directory.parent != dataset_root:
        raise SmokeTrainingError("selection directory must be directly beneath the dataset root")
    selection: dict[str, dict[str, object]] = {}
    for split in ("train", "val"):
        path = selection_directory / f"{split}.txt"
        if path.is_symlink() or not path.is_file():
            raise SmokeTrainingError(f"selection file must be a regular non-symlink file: {path}")
        try:
            resolved_path = path.resolve(strict=True)
        except OSError as exc:
            raise SmokeTrainingError(f"selection file does not exist: {path}") from exc
        if resolved_path.parent != resolved_selection_directory:
            raise SmokeTrainingError(
                "selection file must be directly beneath the selection directory"
            )
        actual_hash = _sha256(resolved_path)
        if actual_hash != hashes[split]:
            raise SmokeTrainingError(f"selection hash does not match provenance for {split}")
        line_count = len(resolved_path.read_text(encoding="utf-8").splitlines())
        if counts is not None and split in counts:
            expected_count = counts[split]
            if type(expected_count) is not int or expected_count != line_count:
                raise SmokeTrainingError(f"selection count does not match provenance for {split}")
        selection[split] = {"path": str(resolved_path), "sha256": actual_hash}
    return selection


def _read_metrics(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise SmokeTrainingError("validation results_dict must be a non-empty mapping")
    metrics: dict[str, float] = {}
    for key, number in value.items():
        if type(key) is not str or not key:
            raise SmokeTrainingError("validation metric names must be non-empty strings")
        if isinstance(number, bool) or not isinstance(number, Real) or not math.isfinite(number):
            raise SmokeTrainingError("validation metrics must be finite numeric values")
        metrics[key] = float(number)
    return metrics


def _weights_are_nonempty(path: Path) -> bool:
    try:
        return not path.is_symlink() and path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def peak_cuda_memory_bytes() -> int:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        allocated = torch.cuda.max_memory_allocated()
        return allocated if type(allocated) is int else 0
    except Exception:
        return 0


def _installed_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _environment() -> dict[str, object]:
    result: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": platform.processor() or "unknown",
        "versions": {
            "torch": _installed_version("torch"),
            "torchvision": _installed_version("torchvision"),
            "ultralytics": _installed_version("ultralytics"),
        },
    }
    try:
        import torch

        available = torch.cuda.is_available()
        cuda: dict[str, object] = {"available": available}
        if available:
            cuda["device_name"] = torch.cuda.get_device_name(0)
        result["cuda"] = cuda
    except Exception as exc:
        result["cuda"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return result


def _git_provenance() -> tuple[str, str | None]:
    repository = Path(__file__).resolve().parents[3]
    try:
        completed = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={repository}",
                "-C",
                str(repository),
                "rev-parse",
                "HEAD",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=10,
        )
        commit = completed.stdout.strip()
        if commit:
            return commit, None
        return "unknown", "git returned an empty commit"
    except (OSError, subprocess.SubprocessError) as exc:
        return "unknown", f"{type(exc).__name__}: {exc}"


read_dataset_provenance = _read_dataset_provenance
read_finite_metrics = _read_metrics
weights_are_nonempty = _weights_are_nonempty
environment_snapshot = _environment
git_provenance = _git_provenance


def _default_model_factory(weights: str) -> object:
    from ultralytics import YOLO

    return YOLO(weights)


def _cleanup_completion_artifacts(completion_path: Path, temporary_path: Path) -> list[Exception]:
    cleanup_errors: list[Exception] = []
    for path in (completion_path, temporary_path):
        try:
            _unlink_file_if_present(path)
        except Exception as exc:
            cleanup_errors.append(exc)
    return cleanup_errors


def _raise_after_cleanup(
    original_error: Exception,
    *,
    completion_path: Path,
    temporary_path: Path,
) -> None:
    cleanup_errors = _cleanup_completion_artifacts(completion_path, temporary_path)
    if cleanup_errors:
        raise SmokeTrainingCleanupError(original_error, cleanup_errors) from original_error
    if isinstance(original_error, SmokeTrainingError):
        raise original_error
    raise SmokeTrainingError("smoke training failed") from original_error


def run_smoke_training(
    config: SmokeTrainingConfig,
    *,
    model_factory: Callable[[str], object] | None = None,
) -> SmokeTrainingResult:
    """Run training and publish completion only after all validation gates pass."""
    completion_path = config.output_root / "completion.json"
    temporary_path = _temporary_path(completion_path)
    initial_cleanup_errors = _cleanup_completion_artifacts(completion_path, temporary_path)
    if initial_cleanup_errors:
        initial_error = SmokeTrainingError("could not clear prior completion artifacts")
        raise SmokeTrainingCleanupError(initial_error, initial_cleanup_errors) from initial_error
    started_at = time.perf_counter()
    try:
        dataset_provenance = read_dataset_provenance(config.data)
        try:
            prediction_source = config.prediction_source.resolve(strict=True)
        except OSError as exc:
            raise SmokeTrainingError(
                f"prediction source directory does not exist: {config.prediction_source}"
            ) from exc
        if prediction_source != dataset_provenance["val_directory"]:
            raise SmokeTrainingError(
                "prediction source must exactly match the data YAML val directory"
            )
        factory = _default_model_factory if model_factory is None else model_factory
        model = factory(config.model)
        model.train(**config.train_kwargs())
        trainer = getattr(model, "trainer", None)
        trainer_save_dir = getattr(trainer, "save_dir", None)
        if trainer_save_dir is None:
            raise SmokeTrainingError("training model did not expose trainer.save_dir")
        try:
            save_dir = Path(trainer_save_dir)
        except TypeError as exc:
            raise SmokeTrainingError("training model has an invalid trainer.save_dir") from exc
        best_weights = save_dir / "weights" / "best.pt"
        last_weights = save_dir / "weights" / "last.pt"
        if not weights_are_nonempty(best_weights) or not weights_are_nonempty(last_weights):
            raise SmokeTrainingError("training did not save non-empty best.pt and last.pt")
        validation = model.val(
            data=str(config.data),
            split="val",
            imgsz=config.imgsz,
            batch=config.batch,
            device=config.device,
            workers=0,
        )
        metrics = read_finite_metrics(validation.results_dict)
        reloaded = factory(str(best_weights))
        predictions = reloaded.predict(
            source=str(config.prediction_source),
            save=True,
            project=str(config.output_root),
            name="reload_prediction",
        )
        if (
            not isinstance(predictions, Sequence)
            or isinstance(predictions, str | bytes | bytearray)
            or not predictions
        ):
            raise SmokeTrainingError("reloaded model produced no prediction results")
        git_commit, git_error = git_provenance()
        payload: dict[str, object] = {
            "status": "completed",
            "experiment_group": config.experiment_group,
            "config": config.to_dict(),
            "best_weights": str(best_weights.resolve()),
            "last_weights": str(last_weights.resolve()),
            "metrics": metrics,
            "elapsed_seconds": time.perf_counter() - started_at,
            "peak_cuda_memory_bytes": peak_cuda_memory_bytes(),
            "git_commit": git_commit,
            "environment": environment_snapshot(),
            "dataset_provenance": dataset_provenance["content"],
            "dataset_provenance_path": dataset_provenance["path"],
            "dataset_provenance_sha256": dataset_provenance["sha256"],
            "data_yaml_sha256": dataset_provenance["data_yaml_sha256"],
            "selected_train_sha256": dataset_provenance["selection_sha256"]["train"],
            "selected_val_sha256": dataset_provenance["selection_sha256"]["val"],
            "selection_files": dataset_provenance["selection_files"],
        }
        if git_error is not None:
            payload["git_error"] = git_error
        write_json_atomic(completion_path, payload)
        return SmokeTrainingResult(
            best_weights=best_weights,
            last_weights=last_weights,
            completion_path=completion_path,
            metrics=metrics,
        )
    except Exception as exc:
        _raise_after_cleanup(
            exc,
            completion_path=completion_path,
            temporary_path=temporary_path,
        )
