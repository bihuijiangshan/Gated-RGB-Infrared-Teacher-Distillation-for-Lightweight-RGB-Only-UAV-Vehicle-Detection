"""Prepare a deterministic, shared-label RGB/IR YOLO smoke dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path

import yaml
from PIL import Image

from rgbt_uav.data.manifest import SampleRecord, manifest_sha256, read_jsonl
from rgbt_uav.data.subset import records_sha256, select_deterministic_records
from rgbt_uav.data.yolo_export import export_yolo_record, format_yolo_labels

_CLASSES = ["car", "truck", "bus", "van", "freight_car"]
_CROP = {"top": 100, "right": 100, "bottom": 100, "left": 100}
_OUTPUT_SIZE = [640, 512]
_REQUIRED_SPLITS = ("train", "val")
_IR_REFERENCE_FILENAMES = {
    "train": "train_ir.jsonl",
    "val": "val_ir.jsonl",
    "test": "test_ir.jsonl",
}

class PrepareYoloSubsetPublicationError(RuntimeError):
    """Raised when a failed publication cannot restore the old final dataset."""

    def __init__(
        self,
        *,
        publication_error: Exception,
        restore_error: Exception,
        backup: Path,
        staging: Path,
    ) -> None:
        self.publication_error = publication_error
        self.restore_error = restore_error
        self.backup = backup
        self.staging = staging
        super().__init__("YOLO subset publication and restoration both failed")


def _resolved(path: Path) -> Path:
    try:
        return path.resolve(strict=False)
    except OSError as exc:
        raise ValueError(f"could not resolve path: {path}") from exc


def _is_strict_child(path: Path, root: Path) -> bool:
    """Return whether a resolved path is beneath, but not equal to, a resolved root."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path != root


def _allowed_output_roots() -> tuple[Path, Path]:
    repository = Path(__file__).resolve().parents[3]
    return _resolved(repository / "data" / "processed"), _resolved(Path(tempfile.gettempdir()))


def _overlaps_forbidden_root(target: Path, forbidden_roots: Sequence[Path]) -> bool:
    return any(
        target == root or _is_strict_child(target, root) or _is_strict_child(root, target)
        for root in forbidden_roots
    )


def _safe_output_target(target: Path, *, forbidden_roots: Sequence[Path]) -> Path:
    resolved = _resolved(target)
    if not any(_is_strict_child(resolved, root) for root in _allowed_output_roots()):
        raise ValueError(
            "output must be a strict child of data/processed or the system temp root: "
            f"{target}"
        )
    if _overlaps_forbidden_root(resolved, forbidden_roots):
        raise ValueError(
            f"output must not overlap a source manifest or raw-data directory: {target}"
        )
    return resolved


def _remove_verified_directory(target: Path, *, forbidden_roots: Sequence[Path]) -> None:
    resolved = _safe_output_target(target, forbidden_roots=forbidden_roots)
    if not resolved.exists():
        return
    if not resolved.is_dir():
        raise ValueError(f"refusing to remove non-directory output target: {resolved}")
    shutil.rmtree(resolved)


def _local_id(index: int, *, count: int) -> str:
    return f"{index:0{max(4, len(str(count)))}d}"


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def _write_yaml(
    path: Path,
    *,
    modality_root: Path,
    splits: Sequence[str],
) -> None:
    payload: dict[str, object] = {
        "path": str(_resolved(modality_root)),
    }
    payload.update(
        {
            split: f"images/{split}"
            for split in splits
        }
    )
    payload.update(
        {
            "nc": len(_CLASSES),
            "names": _CLASSES,
        }
    )
    _write_text(
        path,
        yaml.safe_dump(
            payload,
            allow_unicode=True,
            sort_keys=False,
        ),
    )


def _provenance(
    *,
    manifest_paths: dict[str, Path],
    selected: dict[str, list[SampleRecord]],
    counts: dict[str, int],
    seed: int,
) -> dict[str, object]:
    return {
        "classes": _CLASSES,
        "counts": counts,
        "crop": _CROP,
        "manifests": {
            split: {"path": str(path), "sha256": manifest_sha256(path)}
            for split, path in manifest_paths.items()
        },
        "manifest_naming_contract": {
            split: _IR_REFERENCE_FILENAMES[split]
            for split in manifest_paths
        },
        "output_size": _OUTPUT_SIZE,
        "reference_annotations": "ir",
        "seed": seed,
        "selection_sha256": {
            split: records_sha256(records)
            for split, records in selected.items()
        },
    }


def _export_split(staging: Path, *, split: str, records: Sequence[SampleRecord]) -> None:
    selection = "\n".join(record.pair_id for record in records) + "\n"
    _write_text(staging / "selection" / f"{split}.txt", selection)
    for index, record in enumerate(records, 1):
        local_id = _local_id(index, count=len(records))
        export_yolo_record(
            record,
            modality="rgb",
            image_path=staging / "rgb" / "images" / split / f"{local_id}.jpg",
            label_path=staging / "rgb" / "labels" / split / f"{local_id}.txt",
        )
        export_yolo_record(
            record,
            modality="ir",
            image_path=staging / "ir" / "images" / split / f"{local_id}.png",
            label_path=staging / "ir" / "labels" / split / f"{local_id}.txt",
        )


def _validate_staging(
    staging: Path,
    *,
    final_output: Path,
    manifest_paths: dict[str, Path],
    selected: dict[str, list[SampleRecord]],
    counts: dict[str, int],
    seed: int,
) -> None:
    expected_files = {
        "rgb.yaml",
        "ir.yaml",
        "provenance.json",
        *{
            f"selection/{split}.txt"
            for split in selected
        },
    }
    for split in selected:
        expected_records = selected[split]
        expected_stems = {
            _local_id(index, count=len(expected_records))
            for index in range(1, len(expected_records) + 1)
        }
        selection_path = staging / "selection" / f"{split}.txt"
        expected_selection = "\n".join(record.pair_id for record in expected_records) + "\n"
        if selection_path.read_text(encoding="utf-8") != expected_selection:
            raise ValueError(f"selection order is invalid for {split}")
        for index, record in enumerate(expected_records, 1):
            stem = _local_id(index, count=len(expected_records))
            expected_label = format_yolo_labels(
                record.boxes,
                record.labels,
                width=_OUTPUT_SIZE[0],
                height=_OUTPUT_SIZE[1],
            ).encode("utf-8")
            for modality in ("rgb", "ir"):
                label_path = staging / modality / "labels" / split / f"{stem}.txt"
                if label_path.read_bytes() != expected_label:
                    raise ValueError(f"invalid {modality} labels for {split}/{stem}")
        for modality, extension, mode in (("rgb", ".jpg", "RGB"), ("ir", ".png", "L")):
            image_directory = staging / modality / "images" / split
            label_directory = staging / modality / "labels" / split
            image_stems = {path.stem for path in image_directory.glob(f"*{extension}")}
            label_stems = {path.stem for path in label_directory.glob("*.txt")}
            if image_stems != expected_stems or label_stems != expected_stems:
                raise ValueError(f"unexpected {modality} files for {split}")
            for stem in expected_stems:
                with Image.open(image_directory / f"{stem}{extension}") as image:
                    if image.size != tuple(_OUTPUT_SIZE) or image.mode != mode:
                        raise ValueError(f"invalid {modality} image for {split}/{stem}")
                expected_files.add(f"{modality}/images/{split}/{stem}{extension}")
                expected_files.add(f"{modality}/labels/{split}/{stem}.txt")
    for modality in ("rgb", "ir"):
        actual_yaml = yaml.safe_load((staging / f"{modality}.yaml").read_text(encoding="utf-8"))
        expected_yaml: dict[str, object] = {
            "path": str(_resolved(final_output / modality)),
        }
        expected_yaml.update({split: f"images/{split}" for split in selected})
        expected_yaml.update(
            {
                "nc": len(_CLASSES),
                "names": _CLASSES,
            }
        )
        if actual_yaml != expected_yaml:
            raise ValueError(f"invalid {modality}.yaml")
    actual_provenance = json.loads((staging / "provenance.json").read_text(encoding="utf-8"))
    expected_provenance = _provenance(
        manifest_paths=manifest_paths,
        selected=selected,
        counts=counts,
        seed=seed,
    )
    if actual_provenance != expected_provenance:
        raise ValueError("invalid provenance")
    actual_files = {
        path.relative_to(staging).as_posix() for path in staging.rglob("*") if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError("staging tree contains missing or unexpected files")


def _validate_selected_labels(selected: dict[str, list[SampleRecord]]) -> None:
    for split, records in selected.items():
        for record in records:
            for label in record.labels:
                if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < len(
                    _CLASSES
                ):
                    raise ValueError(
                        f"invalid class label for {split}/{record.pair_id}: {label!r}; "
                        "expected 0..4"
                    )


def _validate_raw_manifest_labels(path: Path) -> None:
    """Reject JSON booleans before ``SampleRecord.from_dict`` coerces them to integers."""
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path} on line {line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"manifest record must be an object in {path} on line {line_number}"
                )
            labels = value.get("labels")
            if not isinstance(labels, list):
                raise ValueError(f"labels must be a list in {path} on line {line_number}")
            for label in labels:
                if type(label) is not int or not 0 <= label < len(_CLASSES):
                    raise ValueError(
                        f"labels must contain class IDs 0..4 in {path} on line {line_number}"
                    )


def _publish_staging(
    *,
    staging: Path,
    final_output: Path,
    backup: Path,
    forbidden_roots: Sequence[Path],
) -> None:
    if not final_output.exists():
        staging.replace(final_output)
        return
    final_output.replace(backup)
    try:
        staging.replace(final_output)
    except Exception as publication_error:
        try:
            backup.replace(final_output)
        except Exception as restore_error:
            raise PrepareYoloSubsetPublicationError(
                publication_error=publication_error,
                restore_error=restore_error,
                backup=backup,
                staging=staging,
            ) from publication_error
        try:
            _remove_verified_directory(staging, forbidden_roots=forbidden_roots)
        except OSError:
            pass
        raise
    _remove_verified_directory(backup, forbidden_roots=forbidden_roots)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a deterministic shared-label RGB/IR YOLO subset"
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--val-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--test-manifest",
        type=Path,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--train-count",
        type=int,
        default=900,
    )
    parser.add_argument(
        "--val-count",
        type=int,
        default=74,
    )
    parser.add_argument(
        "--test-count",
        type=int,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    args = parser.parse_args(argv)

    test_manifest_supplied = (
        args.test_manifest is not None
    )
    test_count_supplied = (
        args.test_count is not None
    )
    if test_manifest_supplied != test_count_supplied:
        raise ValueError(
            "--test-manifest and --test-count "
            "must be provided together"
        )

    manifest_paths = {
        "train": _resolved(args.train_manifest),
        "val": _resolved(args.val_manifest),
    }
    counts = {
        "train": args.train_count,
        "val": args.val_count,
    }

    if (
        args.test_manifest is not None
        and args.test_count is not None
    ):
        manifest_paths["test"] = _resolved(
            args.test_manifest
        )
        counts["test"] = args.test_count

    active_splits = tuple(manifest_paths)
    if active_splits[:2] != _REQUIRED_SPLITS:
        raise ValueError(
            "train and val splits are required"
        )

    for split, path in manifest_paths.items():
        if path.name != _IR_REFERENCE_FILENAMES[split]:
            raise ValueError(
                "IR-reference naming contract requires "
                f"{_IR_REFERENCE_FILENAMES[split]!r} "
                f"for {split}, got {path.name!r}"
            )
        _validate_raw_manifest_labels(path)

    records = {
        split: read_jsonl(path)
        for split, path in manifest_paths.items()
    }
    selected = {
        split: select_deterministic_records(
            records[split],
            count=counts[split],
            seed=args.seed,
            split=split,
        )
        for split in active_splits
    }
    _validate_selected_labels(selected)
    forbidden_roots = [path.parent for path in manifest_paths.values()]
    forbidden_roots.extend(
        _resolved(Path(path)).parent
        for records_for_split in records.values()
        for record in records_for_split
        for path in (record.rgb_path, record.ir_path)
    )
    final_output = _safe_output_target(args.output, forbidden_roots=forbidden_roots)
    staging = _safe_output_target(
        final_output.with_name(f"{final_output.name}.staging"), forbidden_roots=forbidden_roots
    )
    backup = _safe_output_target(
        final_output.with_name(f"{final_output.name}.backup"), forbidden_roots=forbidden_roots
    )
    if final_output.exists() and not final_output.is_dir():
        raise ValueError(f"output exists but is not a directory: {final_output}")
    if final_output.exists() and not args.overwrite:
        raise ValueError(f"output already exists: {final_output}")
    if backup.exists():
        raise ValueError(f"recovery backup already exists: {backup}")
    if staging.exists():
        if not args.overwrite:
            raise ValueError(f"stale staging directory exists: {staging}")
        _remove_verified_directory(staging, forbidden_roots=forbidden_roots)

    try:
        staging.mkdir(parents=True, exist_ok=False)
        for split in active_splits:
            _export_split(
                staging,
                split=split,
                records=selected[split],
            )
        _write_yaml(
            staging / "rgb.yaml",
            modality_root=final_output / "rgb",
            splits=active_splits,
        )
        _write_yaml(
            staging / "ir.yaml",
            modality_root=final_output / "ir",
            splits=active_splits,
        )
        _write_text(
            staging / "provenance.json",
            json.dumps(
                _provenance(
                    manifest_paths=manifest_paths,
                    selected=selected,
                    counts=counts,
                    seed=args.seed,
                ),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        _validate_staging(
            staging,
            final_output=final_output,
            manifest_paths=manifest_paths,
            selected=selected,
            counts=counts,
            seed=args.seed,
        )
        _publish_staging(
            staging=staging,
            final_output=final_output,
            backup=backup,
            forbidden_roots=forbidden_roots,
        )
    except PrepareYoloSubsetPublicationError:
        raise
    except Exception:
        if staging.exists():
            _remove_verified_directory(staging, forbidden_roots=forbidden_roots)
        raise


if __name__ == "__main__":
    main()
