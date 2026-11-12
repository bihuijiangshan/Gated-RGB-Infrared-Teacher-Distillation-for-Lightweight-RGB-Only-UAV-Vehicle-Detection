from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path
from typing import Literal, Protocol

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class PairedDatasetError(ValueError):
    """Raised when RGB and IR samples cannot be paired safely."""

class PairedBatchError(ValueError):
    """Raised when a paired RGB-IR batch is invalid."""


class PairedTransform(Protocol):
    def __call__(
        self,
        rgb: Tensor,
        ir: Tensor,
        boxes: Tensor,
        *,
        pair_id: str,
        split: str,
    ) -> tuple[Tensor, Tensor, Tensor]:
        ...

class PairedYoloDataset(Dataset[dict[str, object]]):
    """Load aligned RGB-IR images and one shared YOLO label file per pair."""

    _IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(
            self,
            data_root: Path,
            *,
            split: Literal["train", "val"],
            modality: Literal["rgb_ir"] = "rgb_ir",
            transform: PairedTransform | None = None,
    ) -> None:
        if modality != "rgb_ir":
            raise PairedDatasetError(f"unsupported modality: {modality!r}")
        if split not in {"train", "val"}:
            raise PairedDatasetError(f"unsupported split: {split!r}")

        self.data_root = Path(data_root)
        self.split = split
        self.modality = modality
        self.transform = transform

        self.rgb_image_root = self.data_root / "rgb" / "images" / split
        self.ir_image_root = self.data_root / "ir" / "images" / split
        self.rgb_label_root = self.data_root / "rgb" / "labels" / split
        self.ir_label_root = self.data_root / "ir" / "labels" / split

        self._validate_directories()

        rgb_images = self._index_images(self.rgb_image_root)
        ir_images = self._index_images(self.ir_image_root)

        if set(rgb_images) != set(ir_images):
            missing_in_ir = sorted(set(rgb_images) - set(ir_images))
            missing_in_rgb = sorted(set(ir_images) - set(rgb_images))
            raise PairedDatasetError(
                "RGB/IR image stems do not match: "
                f"missing_in_ir={missing_in_ir[:5]}, "
                f"missing_in_rgb={missing_in_rgb[:5]}"
            )

        self.samples: list[tuple[str, Path, Path, Path]] = []

        for pair_id in sorted(rgb_images):
            rgb_label = self.rgb_label_root / f"{pair_id}.txt"
            ir_label = self.ir_label_root / f"{pair_id}.txt"

            if not rgb_label.is_file():
                raise PairedDatasetError(f"RGB labels missing for pair {pair_id!r}")
            if not ir_label.is_file():
                raise PairedDatasetError(f"IR labels missing for pair {pair_id!r}")

            if rgb_label.read_bytes() != ir_label.read_bytes():
                raise PairedDatasetError(
                    f"RGB and IR labels differ for pair {pair_id!r}"
                )

            self.samples.append(
                (
                    pair_id,
                    rgb_images[pair_id],
                    ir_images[pair_id],
                    rgb_label,
                )
            )

        if not self.samples:
            raise PairedDatasetError(
                f"no paired samples found for split {split!r} under {self.data_root}"
            )

    def _validate_directories(self) -> None:
        directories = (
            ("RGB image", self.rgb_image_root),
            ("IR image", self.ir_image_root),
            ("RGB label", self.rgb_label_root),
            ("IR label", self.ir_label_root),
        )
        for label, path in directories:
            if not path.is_dir():
                raise PairedDatasetError(f"{label} directory does not exist: {path}")

    @classmethod
    def _index_images(cls, root: Path) -> dict[str, Path]:
        indexed: dict[str, Path] = {}

        for path in sorted(root.iterdir()):
            if not path.is_file() or path.suffix.lower() not in cls._IMAGE_SUFFIXES:
                continue

            pair_id = path.stem
            if pair_id in indexed:
                raise PairedDatasetError(
                    f"multiple image files found for pair {pair_id!r} under {root}"
                )

            indexed[pair_id] = path

        return indexed

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        pair_id, rgb_path, ir_path, label_path = self.samples[index]

        rgb = self._load_image(rgb_path, mode="RGB")
        ir = self._load_image(ir_path, mode="L")

        if rgb.shape[1:] != ir.shape[1:]:
            raise PairedDatasetError(
                f"RGB and IR shapes differ for pair {pair_id!r}: "
                f"{tuple(rgb.shape)} vs {tuple(ir.shape)}"
            )

        boxes, labels = self._load_labels(label_path)

        if self.transform is not None:
            rgb, ir, boxes = self.transform(
                rgb,
                ir,
                boxes,
                pair_id=pair_id,
                split=self.split,
            )
            self._validate_transformed_sample(
                pair_id=pair_id,
                rgb=rgb,
                ir=ir,
                boxes=boxes,
                labels=labels,
            )

        return {
            "rgb": rgb,
            "ir": ir,
            "boxes": boxes,
            "labels": labels,
            "pair_id": pair_id,
            "split": self.split,
        }

    @staticmethod
    def _validate_transformed_sample(
            *,
            pair_id: str,
            rgb: Tensor,
            ir: Tensor,
            boxes: Tensor,
            labels: Tensor,
    ) -> None:
        if rgb.ndim != 3 or ir.ndim != 3:
            raise PairedDatasetError(
                f"images must remain CHW tensors after transform for pair {pair_id!r}"
            )

        if rgb.shape[0] != 3 or ir.shape[0] != 1:
            raise PairedDatasetError(f"image channels changed after transform for pair {pair_id!r}")

        if rgb.shape[1:] != ir.shape[1:]:
            raise PairedDatasetError(
                f"RGB and IR spatial shapes differ after transform for pair {pair_id!r}"
            )

        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise PairedDatasetError(
                f"boxes must remain [num_boxes, 4] after transform for pair {pair_id!r}"
            )

        if boxes.shape[0] != labels.shape[0]:
            raise PairedDatasetError(
                f"box and label counts differ after transform for pair {pair_id!r}"
            )

        if not torch.isfinite(rgb).all():
            raise PairedDatasetError(
                f"RGB contains non-finite values after transform for pair {pair_id!r}"
            )

        if not torch.isfinite(ir).all():
            raise PairedDatasetError(
                f"IR contains non-finite values after transform for pair {pair_id!r}"
            )

        if not torch.isfinite(boxes).all():
            raise PairedDatasetError(
                f"boxes contain non-finite values after transform for pair {pair_id!r}"
            )

        if boxes.numel() > 0 and (torch.any(boxes < 0.0) or torch.any(boxes > 1.0)):
            raise PairedDatasetError(f"boxes leave [0, 1] after transform for pair {pair_id!r}")

    @staticmethod
    def _load_image(path: Path, *, mode: Literal["RGB", "L"]) -> Tensor:
        try:
            with Image.open(path) as image:
                converted = image.convert(mode)
                values = torch.from_numpy(
                    __import__("numpy").asarray(converted, dtype="float32")
                )
        except OSError as exc:
            raise PairedDatasetError(f"could not read image: {path}") from exc

        if mode == "RGB":
            return values.permute(2, 0, 1).contiguous() / 255.0

        return values.unsqueeze(0).contiguous() / 255.0

    @staticmethod
    def _load_labels(path: Path) -> tuple[Tensor, Tensor]:
        boxes: list[list[float]] = []
        labels: list[int] = []

        for line_number, raw_line in enumerate(
                path.read_text(encoding="utf-8").splitlines(),
                start=1,
        ):
            line = raw_line.strip()
            if not line:
                continue

            tokens = line.split()
            if len(tokens) != 5:
                raise PairedDatasetError(
                    f"label must contain 5 tokens at {path}:{line_number}"
                )

            try:
                class_id = int(tokens[0])
                coordinates = [float(token) for token in tokens[1:]]
            except ValueError as exc:
                raise PairedDatasetError(
                    f"label contains non-numeric values at {path}:{line_number}"
                ) from exc

            if class_id < 0:
                raise PairedDatasetError(
                    f"class ID must be non-negative at {path}:{line_number}"
                )

            if any(not math.isfinite(value) for value in coordinates):
                raise PairedDatasetError(
                    f"label contains non-finite coordinates at {path}:{line_number}"
                )

            x_center, y_center, width, height = coordinates
            if width <= 0 or height <= 0:
                raise PairedDatasetError(
                    f"box width and height must be positive at {path}:{line_number}"
                )

            tolerance = 1e-5
            if any(
                    value < -tolerance or value > 1.0 + tolerance
                    for value in coordinates
            ):
                raise PairedDatasetError(
                    f"normalized box is outside [0, 1] at {path}:{line_number}"
                )

            boxes.append(
                [
                    min(max(x_center, 0.0), 1.0),
                    min(max(y_center, 0.0), 1.0),
                    min(max(width, 0.0), 1.0),
                    min(max(height, 0.0), 1.0),
                ]
            )
            labels.append(class_id)

        if not boxes:
            return (
                torch.empty((0, 4), dtype=torch.float32),
                torch.empty((0,), dtype=torch.int64),
            )

        return (
            torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.int64),
        )

    def __iter__(self) -> Iterator[dict[str, object]]:
        for index in range(len(self)):
            yield self[index]

def collate_paired_batch(
    batch: list[dict[str, object]],
) -> dict[str, object]:
    """Stack RGB/IR tensors while preserving variable-length detection targets."""
    if not batch:
        raise PairedBatchError("batch must not be empty")

    first_split = batch[0].get("split")
    if not isinstance(first_split, str):
        raise PairedBatchError("each sample must contain a string split")

    rgb_images: list[Tensor] = []
    ir_images: list[Tensor] = []
    boxes: list[Tensor] = []
    labels: list[Tensor] = []
    pair_ids: list[str] = []

    for index, sample in enumerate(batch):
        split = sample.get("split")
        if split != first_split:
            raise PairedBatchError("all samples in a batch must use the same split")

        rgb = sample.get("rgb")
        ir = sample.get("ir")
        sample_boxes = sample.get("boxes")
        sample_labels = sample.get("labels")
        pair_id = sample.get("pair_id")

        if not isinstance(rgb, Tensor) or not isinstance(ir, Tensor):
            raise PairedBatchError(f"sample {index} must contain tensor images")

        if rgb.ndim != 3 or ir.ndim != 3:
            raise PairedBatchError(
                f"sample {index} images must have shape [channels, height, width]"
            )

        if rgb.shape[1:] != ir.shape[1:]:
            raise PairedBatchError(
                f"sample {index} RGB and IR spatial shapes differ"
            )

        if rgb.shape[0] != 3 or ir.shape[0] != 1:
            raise PairedBatchError(
                f"sample {index} must have RGB channels=3 and IR channels=1"
            )

        if not isinstance(sample_boxes, Tensor) or not isinstance(sample_labels, Tensor):
            raise PairedBatchError(
                f"sample {index} must contain tensor boxes and labels"
            )

        if sample_boxes.ndim != 2 or sample_boxes.shape[1] != 4:
            raise PairedBatchError(
                f"sample {index} boxes must have shape [num_boxes, 4]"
            )

        if sample_labels.ndim != 1:
            raise PairedBatchError(
                f"sample {index} labels must have shape [num_boxes]"
            )

        if sample_boxes.shape[0] != sample_labels.shape[0]:
            raise PairedBatchError(
                f"sample {index} box and label counts differ"
            )

        if not isinstance(pair_id, str) or not pair_id:
            raise PairedBatchError(
                f"sample {index} must contain a non-empty pair_id"
            )

        rgb_images.append(rgb)
        ir_images.append(ir)
        boxes.append(sample_boxes)
        labels.append(sample_labels)
        pair_ids.append(pair_id)

    return {
        "rgb": torch.stack(rgb_images, dim=0),
        "ir": torch.stack(ir_images, dim=0),
        "boxes": boxes,
        "labels": labels,
        "pair_ids": tuple(pair_ids),
        "split": first_split,
    }