from __future__ import annotations

import hashlib
import math
import os
import shutil
import stat
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from tempfile import mkdtemp
from typing import BinaryIO

from rgbt_uav.evaluation.coco_dataset import (
    _BOUNDARY_TOLERANCE_PX,
    CocoDatasetBundle,
    DatasetContractError,
    EvaluationImage,
    InputFingerprint,
    _make_fingerprint,
    _open_verified_binary,
    _read_hashed,
    _require_stable_open_handle,
    _safe_component_observations,
)


class InferenceContractError(ValueError):
    """Raised when inference inputs or results violate the COCO contract."""


@dataclass(frozen=True)
class InferenceSettings:
    imgsz: int = 640
    conf: float = 0.001
    iou: float = 0.7
    max_det: int = 300


@dataclass(frozen=True)
class PredictionEvidence:
    predictions: tuple[dict[str, object], ...]
    images_seen: int
    predictions_seen: int
    settings: InferenceSettings
    weights_path: Path
    weights_sha256: str


@dataclass(frozen=True)
class _WeightSnapshot:
    fingerprint: InputFingerprint
    component_identities: tuple[tuple[Path, int, int, int], ...]


_chmod = os.chmod


def _remove_staging_tree(path: Path) -> None:
    shutil.rmtree(path)


def _uses_protected_original_path() -> bool:
    return os.name == "nt"


def _raise(message: str, *, cause: Exception | None = None) -> None:
    error = InferenceContractError(message)
    if cause is None:
        raise error
    raise error from cause


def _component_identities(path: Path) -> tuple[tuple[Path, int, int, int], ...]:
    return tuple(
        (component, status.st_dev, status.st_ino, stat.S_IFMT(status.st_mode))
        for component, status in _safe_component_observations(
            path, label="model weights"
        )
    )


def _normalized_verified_path(
    components: tuple[tuple[Path, int, int, int], ...],
) -> Path:
    verified_spelling = components[-1][0]
    normalized = Path(os.path.normpath(os.fspath(verified_spelling)))
    verified_anchor = os.path.normcase(os.path.normpath(verified_spelling.anchor))
    normalized_anchor = os.path.normcase(os.path.normpath(normalized.anchor))
    if not normalized.is_absolute() or normalized_anchor != verified_anchor:
        _raise(
            "normalizing model weights changed its verified filesystem root: "
            f"{verified_spelling}"
        )
    return normalized


def _require_verified_leaf_identity(
    *,
    device: int,
    inode: int,
    file_type: int,
    components: tuple[tuple[Path, int, int, int], ...],
    path: Path,
) -> None:
    if (device, inode, file_type) != components[-1][1:]:
        _raise(f"normalized model weights changed verified leaf identity: {path}")


def _snapshot_weights(path: Path, *, verified_path: Path) -> _WeightSnapshot:
    try:
        components_before = _component_identities(path)
        _, fingerprint = _read_hashed(verified_path, label="model weights")
        components_after = _component_identities(path)
    except DatasetContractError as exc:
        _raise(str(exc), cause=exc)
    if components_after != components_before:
        _raise(f"model weights path identity changed while it was inspected: {path}")
    _require_verified_leaf_identity(
        device=fingerprint.device,
        inode=fingerprint.inode,
        file_type=stat.S_IFREG,
        components=components_after,
        path=verified_path,
    )
    return _WeightSnapshot(fingerprint, components_after)


def _read_opened_weights(
    handle: BinaryIO, opened: os.stat_result, *, path: Path
) -> tuple[bytes, InputFingerprint]:
    try:
        handle.seek(0)
        content = handle.read()
    except OSError as exc:
        _raise(f"could not read model weights: {path}", cause=exc)
    try:
        after = _require_stable_open_handle(
            handle, opened, path=path, label="model weights"
        )
    except DatasetContractError as exc:
        _raise(str(exc), cause=exc)
    if len(content) != after.st_size:
        _raise(f"model weights changed while they were read: {path}")
    digest = hashlib.sha256(content).hexdigest()
    return content, _make_fingerprint(path, after, digest)


@contextmanager
def _verified_weight_handle(
    path: Path,
) -> Iterator[tuple[BinaryIO, os.stat_result]]:
    manager = _open_verified_binary(path, label="model weights")
    try:
        opened_handle = manager.__enter__()
    except DatasetContractError as exc:
        _raise(str(exc), cause=exc)
    try:
        yield opened_handle
    except BaseException as primary:
        try:
            manager.__exit__(type(primary), primary, primary.__traceback__)
        except BaseException as cleanup_error:
            _record_cleanup_error(
                primary,
                f"verified checkpoint handle cleanup failed: {cleanup_error}",
            )
        raise
    else:
        try:
            manager.__exit__(None, None, None)
        except DatasetContractError as exc:
            _raise(str(exc), cause=exc)


def _cleanup_staging(directory: Path, staged: Path) -> OSError | None:
    errors: list[OSError] = []

    def restore_permissions() -> None:
        for target, mode in (
            (directory, stat.S_IRWXU),
            (staged, stat.S_IRUSR | stat.S_IWUSR),
        ):
            if not target.exists():
                continue
            try:
                _chmod(target, mode)
            except OSError as exc:
                errors.append(exc)

    restore_permissions()
    try:
        _remove_staging_tree(directory)
    except OSError as exc:
        errors.append(exc)
        restore_permissions()
        try:
            _remove_staging_tree(directory)
        except OSError as retry_exc:
            errors.append(retry_exc)
    if not errors:
        return None
    residual = str(directory) if directory.exists() else "none"
    details = "; ".join(str(error) for error in errors)
    return OSError(
        f"checkpoint staging cleanup failed; residual={residual}; errors={details}"
    )


def _record_cleanup_error(primary: BaseException, detail: str) -> None:
    try:
        primary.checkpoint_cleanup_error = detail
    except (AttributeError, TypeError):
        pass
    add_note = getattr(primary, "add_note", None)
    if callable(add_note):
        add_note(detail)


def _verify_staged_weights(staged: Path, content: bytes) -> None:
    try:
        staged_content, staged_fingerprint = _read_hashed(
            staged, label="staged model weights"
        )
    except DatasetContractError as exc:
        _raise(str(exc), cause=exc)
    if staged_content != content or staged_fingerprint.sha256 != hashlib.sha256(
        content
    ).hexdigest():
        _raise("staged model weights do not match verified bytes")


@contextmanager
def _staged_model_weights(content: bytes, verified_path: Path) -> Iterator[Path]:
    try:
        directory = Path(mkdtemp(prefix="rgbt-uav-weights-"))
    except OSError as exc:
        _raise("could not create a private checkpoint staging directory", cause=exc)
    staged = directory / verified_path.name
    try:
        try:
            staged.write_bytes(content)
            _chmod(staged, stat.S_IRUSR)
            _chmod(directory, stat.S_IRUSR | stat.S_IXUSR)
        except OSError as exc:
            _raise("could not stage verified model weights", cause=exc)
        _verify_staged_weights(staged, content)
        yield staged
    except BaseException as primary:
        cleanup_error = _cleanup_staging(directory, staged)
        if cleanup_error is not None:
            _record_cleanup_error(primary, str(cleanup_error))
        raise
    else:
        cleanup_error = _cleanup_staging(directory, staged)
        if cleanup_error is not None:
            _raise(str(cleanup_error), cause=cleanup_error)


@contextmanager
def _bound_model_weights(
    path: Path,
) -> Iterator[tuple[Path, Path, _WeightSnapshot]]:
    try:
        components_before = _component_identities(path)
    except DatasetContractError as exc:
        _raise(str(exc), cause=exc)
    verified_path = _normalized_verified_path(components_before)
    with _verified_weight_handle(verified_path) as (handle, opened):
        _require_verified_leaf_identity(
            device=opened.st_dev,
            inode=opened.st_ino,
            file_type=stat.S_IFMT(opened.st_mode),
            components=components_before,
            path=verified_path,
        )
        content, fingerprint = _read_opened_weights(
            handle, opened, path=verified_path
        )
        try:
            components_after = _component_identities(path)
        except DatasetContractError as exc:
            _raise(str(exc), cause=exc)
        if components_after != components_before:
            _raise(f"model weights path identity changed while it was bound: {path}")
        snapshot = _WeightSnapshot(fingerprint, components_after)

        if _uses_protected_original_path():
            yield verified_path, verified_path, snapshot
        else:
            with _staged_model_weights(content, verified_path) as staged:
                yield staged, verified_path, snapshot

        after = _snapshot_weights(path, verified_path=verified_path)
        if after != snapshot:
            _raise(f"model weights were replaced, rewritten, or changed: {path}")


def _python_values(value: object, *, label: str) -> object:
    try:
        cpu_value = value.cpu() if hasattr(value, "cpu") else value
        return cpu_value.tolist() if hasattr(cpu_value, "tolist") else cpu_value
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        _raise(f"could not convert result {label} to CPU Python values", cause=exc)


def _sequence(value: object, *, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _raise(f"result {label} must convert to a Python sequence")
    return value


def _real_number(value: object, *, label: str) -> Real:
    if isinstance(value, bool) or not isinstance(value, Real):
        _raise(f"{label} must be numeric and not boolean")
    try:
        finite = value == value and value > -math.inf and value < math.inf
    except (ArithmeticError, TypeError, ValueError) as exc:
        _raise(f"could not validate {label} as a finite number", cause=exc)
    if not finite:
        _raise(f"{label} must be finite")
    return value


def _bounded_float(value: Real, *, label: str) -> float:
    try:
        return float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        _raise(f"could not convert {label} to a native float", cause=exc)


def _class_id(value: object) -> int:
    numeric = _real_number(value, label="class")
    try:
        in_range = 0 <= numeric <= 4
        integral = numeric % 1 == 0
    except (ArithmeticError, TypeError, ValueError) as exc:
        _raise("could not validate class range and integrality", cause=exc)
    if not integral:
        _raise("class must be exactly integral")
    if not in_range:
        _raise("class must be within 0..4")
    try:
        return int(numeric)
    except (OverflowError, TypeError, ValueError) as exc:
        _raise("could not convert class to an integer", cause=exc)


def _confidence(value: object) -> float:
    numeric = _real_number(value, label="confidence")
    try:
        in_range = 0 <= numeric <= 1
    except (ArithmeticError, TypeError, ValueError) as exc:
        _raise("could not validate confidence range", cause=exc)
    if not in_range:
        _raise("confidence must be within [0, 1]")
    return _bounded_float(numeric, label="confidence")


def _normalize_edge(value: float, boundary: float) -> float:
    if abs(value - boundary) <= _BOUNDARY_TOLERANCE_PX:
        return boundary
    return value


def _coco_box(value: object) -> list[float]:
    coordinates = _sequence(value, label="xyxy row")
    if len(coordinates) != 4:
        _raise("each xyxy row must contain exactly four coordinates")
    numeric_left, numeric_top, numeric_right, numeric_bottom = (
        _real_number(coordinate, label="box coordinate")
        for coordinate in coordinates
    )
    tolerance = _BOUNDARY_TOLERANCE_PX
    try:
        positive = numeric_right > numeric_left and numeric_bottom > numeric_top
        in_bounds = (
            numeric_left >= -tolerance
            and numeric_top >= -tolerance
            and numeric_right <= 640.0 + tolerance
            and numeric_bottom <= 512.0 + tolerance
        )
    except (ArithmeticError, TypeError, ValueError) as exc:
        _raise("could not validate box geometry", cause=exc)
    if not positive:
        _raise("box width and height must be positive")
    if not in_bounds:
        _raise("box must lie fully within the native image")
    left, top, right, bottom = (
        _bounded_float(coordinate, label="box coordinate")
        for coordinate in (
            numeric_left,
            numeric_top,
            numeric_right,
            numeric_bottom,
        )
    )
    left = _normalize_edge(left, 0.0)
    top = _normalize_edge(top, 0.0)
    right = _normalize_edge(right, 640.0)
    bottom = _normalize_edge(bottom, 512.0)
    if right <= left or bottom <= top:
        _raise("normalized box width and height must be positive")
    return [left, top, right - left, bottom - top]


def _valid_orig_shape(value: object) -> bool:
    return (
        type(value) is tuple
        and len(value) == 2
        and all(type(dimension) is int for dimension in value)
        and value == (512, 640)
    )


def _resolved_key(path: object, *, label: str) -> str:
    if not isinstance(path, (str, os.PathLike)):
        _raise(f"{label} path must be path-like")
    try:
        return os.path.normcase(str(Path(path).resolve(strict=True)))
    except (OSError, RuntimeError) as exc:
        _raise(f"could not resolve {label} path: {path}", cause=exc)


def _default_model_factory(checkpoint: str) -> object:
    from ultralytics import YOLO

    return YOLO(checkpoint)

def _predict_in_batches(
    predict: Callable[..., Iterator[object]],
    sources: Sequence[str],
    settings: InferenceSettings,
) -> Iterator[object]:
    """Limit each prediction call to at most eight input images."""
    batch_size = 8
    for start in range(0, len(sources), batch_size):
        yield from predict(
            source=list(sources[start : start + batch_size]),
            stream=True,
            imgsz=settings.imgsz,
            conf=settings.conf,
            iou=settings.iou,
            max_det=settings.max_det,
            verbose=False,
        )

def collect_coco_predictions(
    weights: Path,
    bundle: CocoDatasetBundle,
    *,
    model_factory: Callable[[str], object] | None = None,
) -> PredictionEvidence:
    """Collect strict native-coordinate COCO predictions from Ultralytics."""
    settings = InferenceSettings()
    weights_path = Path(weights).absolute()
    with _bound_model_weights(weights_path) as (
        checkpoint,
        original_weights_path,
        before,
    ):
        expected: dict[str, EvaluationImage] = {}
        for image in bundle.images:
            key = _resolved_key(image.path, label="expected image")
            if key in expected:
                _raise(f"duplicate resolved expected image path: {image.path}")
            expected[key] = image

        factory = (
            _default_model_factory if model_factory is None else model_factory
        )
        model = factory(str(checkpoint))
        predict = getattr(model, "predict", None)
        if not callable(predict):
            _raise("model must expose a callable predict method")
        results = _predict_in_batches(
            predict,
            [str(image.path) for image in bundle.images],
            settings,
        )

        by_image: dict[str, list[dict[str, object]]] = {}
        for result in results:
            key = _resolved_key(getattr(result, "path", None), label="result")
            image = expected.get(key)
            if image is None:
                _raise(f"unexpected result path: {getattr(result, 'path', None)}")
            if key in by_image:
                _raise(f"duplicate result path: {getattr(result, 'path', None)}")
            if not _valid_orig_shape(getattr(result, "orig_shape", None)):
                _raise("result orig_shape must be the exact integer pair (512, 640)")

            boxes = getattr(result, "boxes", None)
            converted: list[dict[str, object]] = []
            if boxes is not None:
                xyxy = _sequence(
                    _python_values(getattr(boxes, "xyxy", None), label="xyxy"),
                    label="xyxy",
                )
                confidences = _sequence(
                    _python_values(getattr(boxes, "conf", None), label="conf"),
                    label="conf",
                )
                classes = _sequence(
                    _python_values(getattr(boxes, "cls", None), label="cls"),
                    label="cls",
                )
                if not len(xyxy) == len(confidences) == len(classes):
                    _raise("xyxy, confidence, and class row counts must be equal")
                for coordinates, confidence, class_value in zip(
                    xyxy, confidences, classes, strict=True
                ):
                    converted.append(
                        {
                            "image_id": image.image_id,
                            "category_id": _class_id(class_value),
                            "bbox": _coco_box(coordinates),
                            "score": _confidence(confidence),
                        }
                    )
            by_image[key] = converted

        if set(by_image) != set(expected):
            _raise("inference results are missing expected images")

        predictions = tuple(
            prediction
            for image in bundle.images
            for prediction in by_image[
                _resolved_key(image.path, label="expected image")
            ]
        )
        return PredictionEvidence(
            predictions=predictions,
            images_seen=len(by_image),
            predictions_seen=len(predictions),
            settings=settings,
            weights_path=original_weights_path,
            weights_sha256=before.fingerprint.sha256 or "",
        )
