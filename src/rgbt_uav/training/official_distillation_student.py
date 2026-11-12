"""Official Ultralytics RGB student used for RGB-T distillation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor, nn
from ultralytics.nn.tasks import DetectionModel


class OfficialStudentError(RuntimeError):
    """Raised when an official RGB student cannot be constructed or loaded."""


def _extract_state_dict(
    payload: object,
) -> dict[str, Tensor]:
    """Extract a tensor state dict from common PyTorch/Ultralytics payloads."""

    source: object = payload

    if isinstance(payload, nn.Module):
        source = payload.state_dict()
    elif isinstance(payload, dict):
        for key in ("ema", "model", "state_dict"):
            candidate = payload.get(key)
            if candidate is None:
                continue

            if isinstance(candidate, nn.Module):
                source = candidate.state_dict()
                break

            if isinstance(candidate, dict):
                source = candidate
                break

    if not isinstance(source, dict):
        raise OfficialStudentError(
            "pretrained checkpoint does not contain a state dict"
        )

    state_dict: dict[str, Tensor] = {}

    for key, value in source.items():
        if not isinstance(key, str):
            continue
        if not isinstance(value, Tensor):
            continue

        normalized_key = key
        for prefix in ("module.", "model."):
            if normalized_key.startswith(prefix):
                candidate_key = normalized_key[len(prefix) :]
                if candidate_key.startswith("model."):
                    normalized_key = candidate_key
                elif prefix == "module.":
                    normalized_key = candidate_key

        state_dict[normalized_key] = value.detach()

    if not state_dict:
        raise OfficialStudentError(
            "pretrained checkpoint state dict contains no tensors"
        )

    return state_dict

class OfficialRGBStudent(nn.Module):
    """Expose official YOLO detection features and raw predictions."""

    def __init__(self, model: DetectionModel) -> None:
        super().__init__()
        self.model = model
        self._features: tuple[Tensor, ...] = ()
        self._hook_handle = self.model.model[-1].register_forward_pre_hook(
            self._capture_detect_inputs
        )

    def _capture_detect_inputs(
        self,
        _module: nn.Module,
        inputs: tuple[object, ...],
    ) -> None:
        if len(inputs) != 1:
            raise OfficialStudentError(
                "official Detect head must receive one feature sequence"
            )

        features = inputs[0]
        if not isinstance(features, Sequence):
            raise OfficialStudentError(
                "official Detect input must be a feature sequence"
            )

        tensors = tuple(features)
        if len(tensors) != 3 or not all(
            isinstance(feature, Tensor)
            for feature in tensors
        ):
            raise OfficialStudentError(
                "official student must expose three feature tensors"
            )

        self._features = tensors

    def forward(
        self,
        rgb: Tensor,
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        if not isinstance(rgb, Tensor):
            raise OfficialStudentError(
                "rgb input must be a tensor"
            )
        if rgb.ndim != 4:
            raise OfficialStudentError(
                "rgb input must have shape [B, 3, H, W]"
            )
        if rgb.shape[1] != 3:
            raise OfficialStudentError(
                "rgb input must contain three channels"
            )

        self._features = ()
        output = self.model(rgb)

        if not self._features:
            raise OfficialStudentError(
                "student feature hook did not capture features"
            )

        raw_predictions: object = output
        if (
            isinstance(output, tuple)
            and len(output) == 2
            and isinstance(output[1], Sequence)
        ):
            raw_predictions = output[1]

        if not isinstance(raw_predictions, Sequence):
            raise OfficialStudentError(
                "official student did not return raw prediction levels"
            )

        predictions = tuple(raw_predictions)
        if len(predictions) != 3 or not all(
            isinstance(prediction, Tensor)
            for prediction in predictions
        ):
            raise OfficialStudentError(
                "official student must return three prediction tensors"
            )

        return self._features, predictions

    def __del__(self) -> None:
        hook_handle = getattr(self, "_hook_handle", None)
        if hook_handle is not None:
            hook_handle.remove()


def build_official_rgb_student(
    *,
    model_name: str,
    num_classes: int,
    device: torch.device,
) -> OfficialRGBStudent:
    """Build a trainable three-channel official YOLO student."""

    if not isinstance(model_name, str) or not model_name.strip():
        raise OfficialStudentError(
            "model_name must be a non-empty string"
        )
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise OfficialStudentError(
            "num_classes must be positive"
        )
    if not isinstance(device, torch.device):
        raise OfficialStudentError(
            "device must be a torch.device"
        )

    try:
        model = DetectionModel(
            model_name,
            ch=3,
            nc=num_classes,
            verbose=False,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise OfficialStudentError(
            "failed to build official RGB student"
        ) from exc

    model.nc = num_classes
    model.args = SimpleNamespace(
        box=7.5,
        cls=0.5,
        dfl=1.5,
    )
    model.to(device)
    model.train()

    for parameter in model.parameters():
        parameter.requires_grad_(True)

    return OfficialRGBStudent(model)


def load_official_rgb_student_pretrained(
    student: OfficialRGBStudent,
    checkpoint_path: Path,
) -> dict[str, object]:
    """Load compatible RGB pretrained weights, skipping class head mismatch."""

    if not isinstance(student, OfficialRGBStudent):
        raise OfficialStudentError(
            "student must be an OfficialRGBStudent"
        )

    path = Path(checkpoint_path)
    if not path.is_file():
        raise OfficialStudentError(
            f"pretrained checkpoint must be a regular file: {path}"
        )

    try:
        payload: Any = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
        source = _extract_state_dict(payload)

        target_state = student.model.state_dict()
        compatible: dict[str, Tensor] = {}

        for source_key, value in source.items():
            candidate_keys = [source_key]

            if source_key.startswith("model."):
                candidate_keys.append(source_key.removeprefix("model."))
            else:
                candidate_keys.append(f"model.{source_key}")

            for candidate_key in candidate_keys:
                target = target_state.get(candidate_key)
                if target is not None and target.shape == value.shape:
                    compatible[candidate_key] = value
                    break

        incompatible = student.model.load_state_dict(
            compatible,
            strict=False,
        )
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise OfficialStudentError(
            "RGB pretrained checkpoint is incompatible with student"
        ) from exc

    return {
        "loaded_keys": tuple(sorted(compatible)),
        "missing_keys": tuple(incompatible.missing_keys),
        "unexpected_keys": tuple(incompatible.unexpected_keys),
    }

def _candidate_target_keys(
    source_key: str,
) -> tuple[str, ...]:
    candidates = [source_key]

    if source_key.startswith("model."):
        candidates.append(
            source_key.removeprefix("model.")
        )
    else:
        candidates.append(
            f"model.{source_key}"
        )

    return tuple(dict.fromkeys(candidates))


def _can_slice_width_reduced_tensor(
    source: Tensor,
    target: Tensor,
) -> bool:
    if source.ndim != target.ndim:
        return False

    if source.ndim not in {1, 2, 4}:
        return False

    if not torch.is_floating_point(source):
        return False

    if not torch.is_floating_point(target):
        return False

    if any(
        source_size < target_size
        for source_size, target_size in zip(
            source.shape,
            target.shape,
            strict=True,
        )
    ):
        return False

    if (
        source.ndim == 4
        and source.shape[2:] != target.shape[2:]
    ):
        return False

    return source.shape != target.shape


def _slice_tensor_to_shape(
    source: Tensor,
    target: Tensor,
) -> Tensor:
    slices = tuple(
        slice(0, size)
        for size in target.shape
    )

    return source[slices].detach().clone()


def _is_mismatched_classification_output(
    source: Tensor,
    target: Tensor,
    *,
    num_classes: int,
) -> bool:
    return (
        source.ndim >= 1
        and target.ndim >= 1
        and target.shape[0] == num_classes
        and source.shape[0] != target.shape[0]
    )


def load_official_rgb_student_width_reduced_pretrained(
    student: OfficialRGBStudent,
    checkpoint_path: Path,
) -> dict[str, object]:
    """Initialize a width-reduced student from a wider RGB checkpoint.

    Exact tensors are loaded directly. Floating-point convolution, linear,
    normalization, and bias tensors may be prefix-sliced when every source
    dimension is at least as large as the corresponding target dimension.

    Classification output tensors are never sliced across class count because
    source and target class semantics differ.
    """

    if not isinstance(student, OfficialRGBStudent):
        raise OfficialStudentError(
            "student must be an OfficialRGBStudent"
        )

    path = Path(checkpoint_path)

    if not path.is_file():
        raise OfficialStudentError(
            f"pretrained checkpoint must be a regular file: {path}"
        )

    try:
        payload: Any = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
        source_state = _extract_state_dict(payload)
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise OfficialStudentError(
            "could not read width-reduced pretrained checkpoint"
        ) from exc

    target_state = student.model.state_dict()
    adapted: dict[str, Tensor] = {}
    exact_keys: list[str] = []
    sliced_keys: list[str] = []
    skipped_classification_keys: list[str] = []

    for source_key, source_value in source_state.items():
        for target_key in _candidate_target_keys(
            source_key
        ):
            target_value = target_state.get(target_key)

            if target_value is None:
                continue

            if source_value.shape == target_value.shape:
                adapted[target_key] = source_value.detach()
                exact_keys.append(target_key)
                break

            if _is_mismatched_classification_output(
                source_value,
                target_value,
                num_classes=student.model.nc,
            ):
                skipped_classification_keys.append(
                    target_key
                )
                break

            if _can_slice_width_reduced_tensor(
                source_value,
                target_value,
            ):
                adapted[target_key] = (
                    _slice_tensor_to_shape(
                        source_value,
                        target_value,
                    )
                )
                sliced_keys.append(target_key)
                break

    try:
        incompatible = student.model.load_state_dict(
            adapted,
            strict=False,
        )
    except (
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise OfficialStudentError(
            "width-reduced pretrained state is incompatible"
        ) from exc

    return {
        "loaded_keys": tuple(sorted(adapted)),
        "exact_keys": tuple(sorted(exact_keys)),
        "sliced_keys": tuple(sorted(sliced_keys)),
        "skipped_classification_keys": tuple(
            sorted(
                set(skipped_classification_keys)
            )
        ),
        "missing_keys": tuple(
            incompatible.missing_keys
        ),
        "unexpected_keys": tuple(
            incompatible.unexpected_keys
        ),
    }