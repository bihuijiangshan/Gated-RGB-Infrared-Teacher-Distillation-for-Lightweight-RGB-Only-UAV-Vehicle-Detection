from __future__ import annotations

import hashlib
import math
import os
import re
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import yaml
from PIL import Image, UnidentifiedImageError


class DatasetContractError(ValueError):
    """Raised when a YOLO dataset does not satisfy the evaluation contract."""


@dataclass(frozen=True)
class InputFingerprint:
    path: Path
    device: int
    inode: int
    size: int
    mtime_ns: int
    sha256: str | None


@dataclass(frozen=True)
class EvaluationImage:
    image_id: int
    path: Path
    label_path: Path
    width: int
    height: int


@dataclass(frozen=True)
class CocoDatasetBundle:
    data_path: Path
    data_sha256: str
    class_names: tuple[str, ...]
    images: tuple[EvaluationImage, ...]
    ground_truth: dict[str, object]
    fingerprints: tuple[InputFingerprint, ...]


_CLASS_NAMES = ("car", "truck", "bus", "van", "freight_car")
_IMAGE_SIZE = (640, 512)
_SUPPORTED_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
_CLASS_TOKEN = re.compile(r"[0-9]+\Z", flags=re.ASCII)
_REAL_TOKEN = re.compile(
    r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z",
    flags=re.ASCII,
)
_WINDOWS_REPARSE_ATTRIBUTE = 0x400
_YOLO_SERIALIZATION_DECIMALS = 6
_YOLO_COMPONENT_ROUNDING_NORMALIZED = 0.5 * 10 ** (-_YOLO_SERIALIZATION_DECIMALS)
# Center and size are rounded independently. Reconstructing one edge therefore
# accumulates one center rounding unit plus half one size rounding unit.
_YOLO_EDGE_ROUNDING_NORMALIZED = (
    _YOLO_COMPONENT_ROUNDING_NORMALIZED
    + _YOLO_COMPONENT_ROUNDING_NORMALIZED / 2
)
_BOUNDARY_TOLERANCE_PX = (
    max(_IMAGE_SIZE) * _YOLO_EDGE_ROUNDING_NORMALIZED
    + math.ulp(float(max(_IMAGE_SIZE)))
)


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that also refuses ambiguous duplicate mapping keys."""

    def construct_mapping(self, node: yaml.nodes.MappingNode, deep: bool = False) -> Any:
        self.flatten_mapping(node)
        seen: set[object] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in seen
            except TypeError as exc:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found an unhashable mapping key",
                    key_node.start_mark,
                ) from exc
            if duplicate:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _raise(message: str, *, cause: Exception | None = None) -> None:
    error = DatasetContractError(message)
    if cause is None:
        raise error
    raise error from cause


def _absolute(path: Path) -> Path:
    try:
        return path.absolute()
    except OSError as exc:
        _raise(f"could not make dataset path absolute: {path}", cause=exc)


def _path_components(path: Path) -> tuple[Path, ...]:
    absolute = _absolute(path)
    components: list[Path] = []
    current = Path(absolute.anchor)
    if absolute.anchor:
        components.append(current)
    for part in absolute.parts[1:] if absolute.anchor else absolute.parts:
        current /= part
        components.append(current)
    return tuple(components)


def _is_link_or_reparse(status: os.stat_result) -> bool:
    attributes = getattr(status, "st_file_attributes", 0)
    return stat.S_ISLNK(status.st_mode) or bool(attributes & _WINDOWS_REPARSE_ATTRIBUTE)


def _safe_component_observations(
    path: Path, *, label: str
) -> tuple[tuple[Path, os.stat_result], ...]:
    components = _path_components(path)
    if not components:
        _raise(f"{label} is not a usable path: {path}")
    observations: list[tuple[Path, os.stat_result]] = []
    for component in components:
        try:
            status = os.stat(component, follow_symlinks=False)
        except OSError as exc:
            _raise(f"{label} is missing or inaccessible: {path}", cause=exc)
        if _is_link_or_reparse(status):
            _raise(f"{label} contains a symlink or reparse point: {component}")
        observations.append((component, status))
    return tuple(observations)


def _require_safe_components(path: Path, *, label: str) -> os.stat_result:
    return _safe_component_observations(path, label=label)[-1][1]


def _require_directory(path: Path, *, label: str) -> Path:
    status = _require_safe_components(path, label=label)
    if not stat.S_ISDIR(status.st_mode):
        _raise(f"{label} must be a regular directory: {path}")
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        _raise(f"could not resolve {label}: {path}", cause=exc)


def _require_regular_file(path: Path, *, label: str) -> os.stat_result:
    status = _require_safe_components(path, label=label)
    if not stat.S_ISREG(status.st_mode):
        _raise(f"{label} must be a regular file: {path}")
    return status


def _same_metadata(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev,
        left.st_ino,
        left.st_size,
        left.st_mtime_ns,
    ) == (
        right.st_dev,
        right.st_ino,
        right.st_size,
        right.st_mtime_ns,
    )


def _same_identity_and_type(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino, stat.S_IFMT(left.st_mode)) == (
        right.st_dev,
        right.st_ino,
        stat.S_IFMT(right.st_mode),
    )


def _make_fingerprint(
    path: Path, status: os.stat_result, digest: str | None
) -> InputFingerprint:
    return InputFingerprint(
        path=path,
        device=status.st_dev,
        inode=status.st_ino,
        size=status.st_size,
        mtime_ns=status.st_mtime_ns,
        sha256=digest,
    )


def _close_descriptors_quietly(descriptors: list[int] | tuple[int, ...]) -> None:
    for descriptor in reversed(descriptors):
        try:
            os.close(descriptor)
        except OSError:
            pass


def _posix_anchored_descriptors(path: Path, *, label: str) -> tuple[int, tuple[int, ...]]:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory_flag is None:
        _raise(f"platform cannot anchor {label} without following links: {path}")
    observations = _safe_component_observations(path, label=label)
    if len(observations) < 2:
        _raise(f"{label} must have a filesystem-root parent: {path}")
    directory_flags = (
        os.O_RDONLY | directory_flag | no_follow | getattr(os, "O_CLOEXEC", 0)
    )
    leaf_flags = os.O_RDONLY | no_follow | getattr(os, "O_CLOEXEC", 0)
    opened_directories: list[int] = []
    leaf_descriptor: int | None = None
    try:
        for index, (component, expected) in enumerate(observations):
            is_leaf = index == len(observations) - 1
            try:
                if index == 0:
                    descriptor = os.open(component, directory_flags)
                else:
                    descriptor = os.open(
                        component.name,
                        leaf_flags if is_leaf else directory_flags,
                        dir_fd=opened_directories[-1],
                    )
            except OSError as exc:
                _raise(f"could not anchor opened {label} component: {component}", cause=exc)
            try:
                opened = os.fstat(descriptor)
            except OSError as exc:
                _close_descriptors_quietly([descriptor])
                _raise(f"could not inspect anchored {label} component: {component}", cause=exc)
            expected_directory = not is_leaf
            if expected_directory != stat.S_ISDIR(opened.st_mode):
                _close_descriptors_quietly([descriptor])
                _raise(f"opened {label} component has the wrong type: {component}")
            if not _same_identity_and_type(expected, opened):
                _close_descriptors_quietly([descriptor])
                _raise(f"opened {label} component identity changed: {component}")
            if is_leaf:
                if not stat.S_ISREG(opened.st_mode):
                    _close_descriptors_quietly([descriptor])
                    _raise(f"opened {label} leaf must be a regular file: {component}")
                leaf_descriptor = descriptor
            else:
                opened_directories.append(descriptor)
        assert leaf_descriptor is not None
        return leaf_descriptor, tuple(opened_directories)
    except BaseException:
        if leaf_descriptor is not None:
            _close_descriptors_quietly([leaf_descriptor])
        _close_descriptors_quietly(opened_directories)
        raise


def _windows_anchored_descriptors(
    path: Path, *, label: str
) -> tuple[int, tuple[int, ...]]:
    import ctypes
    import msvcrt
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

    generic_read = 0x80000000
    read_attributes = 0x00000080
    # Keep every anchored component readable by peers while denying concurrent
    # write/delete access for the lifetime of the verified handle chain.
    share_read = 0x00000001
    open_existing = 3
    open_reparse_point = 0x00200000
    backup_semantics = 0x02000000
    file_attribute_normal = 0x00000080
    file_attribute_directory = 0x00000010
    invalid_handle = ctypes.c_void_p(-1).value
    observations = _safe_component_observations(path, label=label)
    if len(observations) < 2:
        _raise(f"{label} must have a filesystem-root parent: {path}")
    opened_directories: list[int] = []
    leaf_descriptor: int | None = None

    def extended_path(component: Path) -> str:
        absolute = str(_absolute(component))
        if absolute.startswith("\\\\?\\"):
            return absolute
        if absolute.startswith("\\\\"):
            return "\\\\?\\UNC\\" + absolute[2:]
        return "\\\\?\\" + absolute

    def open_component(component: Path, *, is_directory: bool) -> int:
        handle = create_file(
            extended_path(component),
            read_attributes if is_directory else generic_read,
            share_read,
            None,
            open_existing,
            file_attribute_normal
            | open_reparse_point
            | (backup_semantics if is_directory else 0),
            None,
        )
        if handle == invalid_handle:
            error = ctypes.get_last_error()
            _raise(
                f"could not anchor opened {label} component: {component}",
                cause=OSError(error),
            )
        information = ByHandleFileInformation()
        if not get_information(handle, ctypes.byref(information)):
            error = ctypes.get_last_error()
            close_handle(handle)
            _raise(
                f"could not inspect opened {label} component: {component}",
                cause=OSError(error),
            )
        opened_is_directory = bool(
            information.file_attributes & file_attribute_directory
        )
        if information.file_attributes & _WINDOWS_REPARSE_ATTRIBUTE:
            close_handle(handle)
            _raise(f"opened {label} component is a reparse point: {component}")
        if opened_is_directory != is_directory:
            close_handle(handle)
            _raise(f"opened {label} component has the wrong type: {component}")
        try:
            return msvcrt.open_osfhandle(
                handle, os.O_RDONLY | getattr(os, "O_BINARY", 0)
            )
        except OSError as exc:
            close_handle(handle)
            _raise(
                f"could not bind opened {label} component: {component}", cause=exc
            )

    try:
        for index, (component, expected) in enumerate(observations):
            is_leaf = index == len(observations) - 1
            descriptor = open_component(component, is_directory=not is_leaf)
            try:
                opened = os.fstat(descriptor)
            except OSError as exc:
                _close_descriptors_quietly([descriptor])
                _raise(f"could not inspect anchored {label} component: {component}", cause=exc)
            if not _same_identity_and_type(expected, opened):
                _close_descriptors_quietly([descriptor])
                _raise(f"opened {label} component identity changed: {component}")
            if is_leaf:
                if not stat.S_ISREG(opened.st_mode):
                    _close_descriptors_quietly([descriptor])
                    _raise(f"opened {label} leaf must be a regular file: {component}")
                leaf_descriptor = descriptor
            else:
                opened_directories.append(descriptor)
        assert leaf_descriptor is not None
        return leaf_descriptor, tuple(opened_directories)
    except BaseException:
        if leaf_descriptor is not None:
            _close_descriptors_quietly([leaf_descriptor])
        _close_descriptors_quietly(opened_directories)
        raise


def _anchored_descriptors(path: Path, *, label: str) -> tuple[int, tuple[int, ...]]:
    if os.name == "nt":
        return _windows_anchored_descriptors(path, label=label)
    return _posix_anchored_descriptors(path, label=label)


@contextmanager
def _open_verified_binary(
    path: Path, *, label: str
) -> Iterator[tuple[BinaryIO, os.stat_result]]:
    expected = _require_regular_file(path, label=label)
    descriptor, directory_descriptors = _anchored_descriptors(path, label=label)
    try:
        handle = os.fdopen(descriptor, "rb")
    except (OSError, ValueError) as exc:
        _close_descriptors_quietly([descriptor, *directory_descriptors])
        _raise(f"could not create a binary stream for {label}: {path}", cause=exc)
    try:
        try:
            opened = os.fstat(handle.fileno())
        except OSError as exc:
            _raise(f"could not inspect anchored {label} leaf: {path}", cause=exc)
        if not stat.S_ISREG(opened.st_mode):
            _raise(f"opened {label} must be a regular file: {path}")
        if not _same_metadata(expected, opened):
            _raise(f"{label} identity changed before its handle was bound: {path}")
        yield handle, opened
    except BaseException:
        try:
            handle.close()
        except OSError:
            pass
        _close_descriptors_quietly(directory_descriptors)
        raise
    else:
        close_error: OSError | None = None
        try:
            handle.close()
        except OSError as exc:
            close_error = exc
        for held_descriptor in reversed(directory_descriptors):
            try:
                os.close(held_descriptor)
            except OSError as exc:
                if close_error is None:
                    close_error = exc
        if close_error is not None:
            _raise(f"could not close anchored handles for {label}: {path}", cause=close_error)


def _require_stable_open_handle(
    handle: BinaryIO, opened: os.stat_result, *, path: Path, label: str
) -> os.stat_result:
    try:
        after = os.fstat(handle.fileno())
    except OSError as exc:
        _raise(f"could not revalidate opened {label}: {path}", cause=exc)
    if not _same_metadata(opened, after):
        _raise(f"opened {label} changed while it was in use: {path}")
    return after


def _read_hashed(path: Path, *, label: str) -> tuple[bytes, InputFingerprint]:
    with _open_verified_binary(path, label=label) as (handle, opened):
        try:
            content = handle.read()
        except OSError as exc:
            _raise(f"could not read {label}: {path}", cause=exc)
        after = _require_stable_open_handle(
            handle, opened, path=path, label=label
        )
        if len(content) != after.st_size:
            _raise(f"{label} changed while it was being read: {path}")
    digest = hashlib.sha256(content).hexdigest()
    return content, _make_fingerprint(path, after, digest)


def _load_yaml(path: Path) -> tuple[dict[str, object], InputFingerprint]:
    content, fingerprint = _read_hashed(path, label="dataset YAML")
    try:
        text = content.decode("utf-8")
        value = yaml.load(text, Loader=_UniqueKeySafeLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        _raise(f"dataset YAML is not valid strict UTF-8 safe YAML: {path}", cause=exc)
    if not isinstance(value, dict):
        _raise("dataset YAML root must be a mapping")
    return value, fingerprint


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _dataset_roots(
    source: Path, data: dict[str, object], split: str
) -> tuple[Path, Path, Path]:
    if split != "val":
        _raise("only the exact 'val' split is supported")
    root_value = data.get("path")
    split_value = data.get("val")
    if type(root_value) is not str or not root_value:
        _raise("dataset YAML path must be a non-empty string")
    if type(split_value) is not str or not split_value:
        _raise("dataset YAML val path must be a non-empty string")

    declared_root = Path(root_value)
    root_candidate = declared_root if declared_root.is_absolute() else source.parent / declared_root
    root = _require_directory(root_candidate, label="dataset root")
    if not declared_root.is_absolute() and not _is_within(root, source.parent):
        _raise("relative dataset root escapes the YAML directory")

    declared_split = Path(split_value)
    if declared_split.is_absolute():
        _raise("dataset split path must be relative to the dataset root")
    image_root = _require_directory(root / declared_split, label="split image root")
    if not _is_within(image_root, root):
        _raise("dataset split path escapes the dataset root")
    relative_split = image_root.relative_to(root)
    if not relative_split.parts or relative_split.parts[0].lower() != "images":
        _raise("dataset split path must be under the images tree")
    label_root = _require_directory(
        root.joinpath("labels", *relative_split.parts[1:]), label="split label root"
    )
    return root, image_root, label_root


def _validate_schema(data: dict[str, object]) -> tuple[str, ...]:
    if type(data.get("nc")) is not int or data["nc"] != 5:
        _raise("dataset nc must be the exact integer 5")
    names = data.get("names")
    if type(names) is not list or tuple(names) != _CLASS_NAMES:
        _raise(f"dataset names must be the exact ordered list {_CLASS_NAMES!r}")
    return tuple(str(name) for name in names)


def _walk_regular_files(root: Path, *, label: str) -> list[Path]:
    files: list[Path] = []

    def enumeration_error(exc: OSError) -> None:
        _raise(f"could not enumerate {label}: {root}", cause=exc)

    try:
        walker = os.walk(
            root, topdown=True, onerror=enumeration_error, followlinks=False
        )
        for directory, directories, filenames in walker:
            directory_path = Path(directory)
            _require_directory(directory_path, label=label)
            for name in directories:
                _require_directory(directory_path / name, label=label)
            for name in filenames:
                path = directory_path / name
                _require_regular_file(path, label=label)
                files.append(path)
    except DatasetContractError:
        raise
    except OSError as exc:
        _raise(f"could not enumerate {label}: {root}", cause=exc)
    return files


def _relative_key(path: Path, root: Path) -> str:
    return path.relative_to(root).with_suffix("").as_posix()


def _paired_inputs(image_root: Path, label_root: Path) -> list[tuple[Path, Path, str]]:
    all_image_files = _walk_regular_files(image_root, label="image input")
    images = [path for path in all_image_files if path.suffix.lower() in _SUPPORTED_SUFFIXES]
    images.sort(key=lambda path: path.relative_to(image_root).as_posix())
    labels = [
        path
        for path in _walk_regular_files(label_root, label="label input")
        if path.suffix.lower() == ".txt"
    ]
    labels.sort(key=lambda path: path.relative_to(label_root).as_posix())

    image_by_key: dict[str, Path] = {}
    image_identities: set[tuple[int, int]] = set()
    resolved_images: set[str] = set()
    for image in images:
        key = _relative_key(image, image_root)
        normalized = os.path.normcase(str(image.resolve(strict=True)))
        status = _require_regular_file(image, label="image input")
        identity = (status.st_dev, status.st_ino)
        if key in image_by_key or normalized in resolved_images or identity in image_identities:
            _raise(f"duplicate resolved image input: {image}")
        image_by_key[key] = image
        resolved_images.add(normalized)
        image_identities.add(identity)

    label_by_key: dict[str, Path] = {}
    for label in labels:
        key = _relative_key(label, label_root)
        if key in label_by_key:
            _raise(f"duplicate label input: {label}")
        label_by_key[key] = label
    missing = sorted(set(image_by_key) - set(label_by_key))
    extra = sorted(set(label_by_key) - set(image_by_key))
    if missing:
        _raise(f"missing label files for images: {missing!r}")
    if extra:
        _raise(f"extra label files without images: {extra!r}")
    return [(image_by_key[key], label_by_key[key], key) for key in sorted(image_by_key)]


def _inspect_image(path: Path) -> tuple[int, int, InputFingerprint]:
    with _open_verified_binary(path, label="image input") as (handle, opened):
        try:
            with Image.open(handle) as image:
                image.verify()
            handle.seek(0)
            with Image.open(handle) as image:
                image.load()
                size = image.size
                mode = image.mode
                image_format = image.format
        except (OSError, SyntaxError, UnidentifiedImageError) as exc:
            _raise(f"image is unreadable or corrupt: {path}", cause=exc)
        after = _require_stable_open_handle(
            handle, opened, path=path, label="image input"
        )
    if size != _IMAGE_SIZE:
        _raise(f"image must be exactly 640 by 512 pixels: {path}")
    if (mode, image_format) not in {("RGB", "JPEG"), ("L", "PNG")}:
        _raise(f"image must be RGB/JPEG or L/PNG: {path}")
    expected_suffixes = {"JPEG": {".jpg", ".jpeg"}, "PNG": {".png"}}
    if path.suffix.lower() not in expected_suffixes[image_format]:
        _raise(f"image filename extension does not match its actual format: {path}")
    return size[0], size[1], _make_fingerprint(path, after, None)


def _normalize_edge(value: float, boundary: float) -> float:
    if abs(value - boundary) <= _BOUNDARY_TOLERANCE_PX:
        return boundary
    return value


def _parse_label(
    content: bytes, *, path: Path, image_id: int, first_annotation_id: int
) -> list[dict[str, object]]:
    try:
        text = content.decode("ascii")
    except UnicodeDecodeError as exc:
        _raise(f"label file must contain ASCII text: {path}", cause=exc)
    if content and not text.strip():
        _raise(f"label file contains only blank whitespace: {path}")

    annotations: list[dict[str, object]] = []
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        if not raw_line.strip():
            continue
        tokens = raw_line.split()
        if len(tokens) != 5:
            _raise(f"{path}:{line_number}: expected exactly five YOLO tokens")
        class_token = tokens[0]
        if _CLASS_TOKEN.fullmatch(class_token) is None:
            _raise(f"{path}:{line_number}: class must be an unsigned base-10 integer")
        try:
            class_id = int(class_token, 10)
        except ValueError as exc:
            _raise(f"{path}:{line_number}: class integer token is too long", cause=exc)
        if not 0 <= class_id < len(_CLASS_NAMES):
            _raise(f"{path}:{line_number}: class ID is outside 0..4")
        if any(_REAL_TOKEN.fullmatch(token) is None for token in tokens[1:]):
            _raise(f"{path}:{line_number}: box tokens must be decimal real numbers")
        try:
            x_center, y_center, box_width, box_height = map(float, tokens[1:])
        except ValueError as exc:
            _raise(f"{path}:{line_number}: box tokens must be real numbers", cause=exc)
        values = (x_center, y_center, box_width, box_height)
        if not all(math.isfinite(value) for value in values):
            _raise(f"{path}:{line_number}: box tokens must be finite")
        if box_width <= 0 or box_height <= 0:
            _raise(f"{path}:{line_number}: box width and height must be positive")

        native_width = box_width * _IMAGE_SIZE[0]
        native_height = box_height * _IMAGE_SIZE[1]
        left = x_center * _IMAGE_SIZE[0] - native_width / 2
        top = y_center * _IMAGE_SIZE[1] - native_height / 2
        right = left + native_width
        bottom = top + native_height
        tolerance = _BOUNDARY_TOLERANCE_PX
        if (
            left < -tolerance
            or top < -tolerance
            or right > _IMAGE_SIZE[0] + tolerance
            or bottom > _IMAGE_SIZE[1] + tolerance
        ):
            _raise(f"{path}:{line_number}: box lies outside the image")
        left = _normalize_edge(left, 0.0)
        top = _normalize_edge(top, 0.0)
        right = _normalize_edge(right, float(_IMAGE_SIZE[0]))
        bottom = _normalize_edge(bottom, float(_IMAGE_SIZE[1]))
        if not all(math.isfinite(value) for value in (left, top, right, bottom)):
            _raise(f"{path}:{line_number}: normalized box coordinates must be finite")
        if not (
            0.0 <= left < right <= float(_IMAGE_SIZE[0])
            and 0.0 <= top < bottom <= float(_IMAGE_SIZE[1])
        ):
            _raise(f"{path}:{line_number}: box lies outside the image after normalization")
        native_width = right - left
        native_height = bottom - top
        area = native_width * native_height
        if (
            not math.isfinite(native_width)
            or not math.isfinite(native_height)
            or not math.isfinite(area)
            or native_width <= 0.0
            or native_height <= 0.0
            or area <= 0.0
        ):
            _raise(f"{path}:{line_number}: normalized box dimensions and area must be positive")
        annotations.append(
            {
                "id": first_annotation_id + len(annotations),
                "image_id": image_id,
                "category_id": class_id,
                "bbox": [left, top, native_width, native_height],
                "area": area,
                "iscrowd": 0,
            }
        )
    return annotations


def load_yolo_coco_dataset(data_path: Path, split: str = "val") -> CocoDatasetBundle:
    """Load one strict project YOLO validation dataset as native-coordinate COCO."""
    source = _absolute(Path(data_path))
    data, data_fingerprint = _load_yaml(source)
    class_names = _validate_schema(data)
    _, image_root, label_root = _dataset_roots(source, data, split)
    pairs = _paired_inputs(image_root, label_root)

    images: list[EvaluationImage] = []
    annotations: list[dict[str, object]] = []
    fingerprints = [data_fingerprint]
    ground_truth_images: list[dict[str, object]] = []
    for image_id, (image_path, label_path, relative_stem) in enumerate(pairs, 1):
        width, height, image_fingerprint = _inspect_image(image_path)
        label_content, label_fingerprint = _read_hashed(label_path, label="label input")
        image = EvaluationImage(image_id, image_path, label_path, width, height)
        images.append(image)
        fingerprints.extend((image_fingerprint, label_fingerprint))
        ground_truth_images.append(
            {
                "id": image_id,
                "file_name": relative_stem + image_path.suffix,
                "width": width,
                "height": height,
            }
        )
        annotations.extend(
            _parse_label(
                label_content,
                path=label_path,
                image_id=image_id,
                first_annotation_id=len(annotations) + 1,
            )
        )

    ground_truth: dict[str, object] = {
        "images": ground_truth_images,
        "annotations": annotations,
        "categories": [
            {"id": category_id, "name": name}
            for category_id, name in enumerate(class_names)
        ],
    }
    return CocoDatasetBundle(
        data_path=source,
        data_sha256=data_fingerprint.sha256 or "",
        class_names=class_names,
        images=tuple(images),
        ground_truth=ground_truth,
        fingerprints=tuple(fingerprints),
    )


def revalidate_dataset(bundle: CocoDatasetBundle) -> None:
    """Confirm every stored dataset input still has its captured no-follow identity."""
    for expected in bundle.fingerprints:
        if expected.sha256 is None:
            with _open_verified_binary(
                expected.path, label="dataset input"
            ) as (handle, opened):
                status = _require_stable_open_handle(
                    handle, opened, path=expected.path, label="dataset input"
                )
                actual = _make_fingerprint(expected.path, status, None)
        else:
            _, actual = _read_hashed(expected.path, label="dataset input")
        if actual != expected:
            _raise(f"dataset input was replaced, rewritten, or changed: {expected.path}")
