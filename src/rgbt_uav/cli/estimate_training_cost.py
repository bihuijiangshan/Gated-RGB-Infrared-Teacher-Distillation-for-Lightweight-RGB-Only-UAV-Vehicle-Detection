from __future__ import annotations

import argparse
import json
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from rgbt_uav.cli.record_training_completion import _load_json_object
from rgbt_uav.training.costing import apply_hourly_price, estimate_gpu_hours


def _unlink_if_owned(path: Path, owned_stat: os.stat_result) -> None:
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
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = (wintypes.HANDLE,)
        close_handle.restype = wintypes.BOOL
        try:
            descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY)
        except BaseException:
            close_handle(handle)
            raise
        try:
            opened_stat = os.fstat(descriptor)
            if not os.path.samestat(opened_stat, owned_stat):
                return
            _mark_open_file_for_deletion_windows(descriptor)
        finally:
            os.close(descriptor)
        return


    _unlink_if_owned_posix(path, owned_stat)


def _unlink_if_owned_posix(path: Path, owned_stat: os.stat_result) -> None:
    """Delete a cooperative-writer artifact after repeated identity checks.

    The sibling publication lock prevents compliant writers from replacing this
    name. POSIX stdlib has no unlink-by-handle, so a process that ignores the lock
    can still race the final identity check and unlink syscall; no stronger claim
    is made here.
    """
    try:
        current_stat = path.lstat()
    except OSError:
        return
    if not stat.S_ISREG(current_stat.st_mode) or not os.path.samestat(
        current_stat, owned_stat
    ):
        return
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        return
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_BINARY", 0) | nofollow,
        )
    except OSError:
        return
    try:
        opened_stat = os.fstat(descriptor)
        if not os.path.samestat(opened_stat, owned_stat):
            return
        final_stat = path.lstat()
        if not stat.S_ISREG(final_stat.st_mode) or not os.path.samestat(
            final_stat, opened_stat
        ):
            return
        path.unlink()
    finally:
        os.close(descriptor)


def _mark_open_file_for_deletion_windows(descriptor: int) -> None:
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
        msvcrt.get_osfhandle(descriptor),
        4,
        ctypes.byref(disposition),
        ctypes.sizeof(disposition),
    ):
        raise ctypes.WinError(ctypes.get_last_error())


def _require_owned_strict_bytes(
    path: Path,
    owned_stat: os.stat_result,
    expected: bytes,
) -> None:
    path_stat = path.lstat()
    if not stat.S_ISREG(path_stat.st_mode) or not os.path.samestat(path_stat, owned_stat):
        raise OSError("JSON output path identity changed")
    with path.open("rb") as stream:
        opened_stat = os.fstat(stream.fileno())
        if not os.path.samestat(opened_stat, owned_stat):
            raise OSError("JSON output handle identity changed")
        if stream.read() != expected:
            raise OSError("JSON output bytes changed")
        ending_stat = os.fstat(stream.fileno())
        if not os.path.samestat(ending_stat, owned_stat):
            raise OSError("JSON output handle identity changed")
        final_path_stat = path.lstat()
        if not stat.S_ISREG(final_path_stat.st_mode) or not os.path.samestat(
            final_path_stat, ending_stat
        ):
            raise OSError("JSON output path identity changed")


def _write_json_atomic_no_clobber(path: Path, payload: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_name(f".{destination.name}.lock")
    expected = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    lock_stat: os.stat_result | None = None
    temporary: Path | None = None
    owned_stat: os.stat_result | None = None
    publication_attempted = False
    try:
        with lock_path.open("x+b") as lock_stream:
            lock_stat = os.fstat(lock_stream.fileno())
            if not os.path.samestat(lock_stat, lock_path.lstat()):
                raise OSError("cost publication lock identity changed")
            lock_stream.write(b"rgbt-uav cost publication cooperative lock\n")
            lock_stream.flush()
            os.fsync(lock_stream.fileno())
            if not os.path.samestat(lock_stat, os.fstat(lock_stream.fileno())):
                raise OSError("cost publication lock identity changed")
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"output path already exists: {destination}")

            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
            )
            temporary = Path(temporary_name)
            with os.fdopen(descriptor, "w+b") as stream:
                owned_stat = os.fstat(stream.fileno())
                path_stat = temporary.lstat()
                if not os.path.samestat(owned_stat, path_stat):
                    raise OSError("temporary output identity changed")
                stream.write(expected)
                stream.flush()
                os.fsync(stream.fileno())
                stream.seek(0)
                if stream.read() != expected:
                    raise OSError("temporary output bytes changed")
                if not os.path.samestat(owned_stat, os.fstat(stream.fileno())):
                    raise OSError("temporary output identity changed")
            _require_owned_strict_bytes(temporary, owned_stat, expected)
            publication_attempted = True
            os.link(temporary, destination, follow_symlinks=False)
            _require_owned_strict_bytes(destination, owned_stat, expected)
            _unlink_if_owned(temporary, owned_stat)
    except BaseException:
        if publication_attempted and owned_stat is not None:
            _unlink_if_owned(destination, owned_stat)
        raise
    finally:
        if temporary is not None and owned_stat is not None:
            _unlink_if_owned(temporary, owned_stat)
        if lock_stat is not None:
            _unlink_if_owned(lock_path, lock_stat)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate full training cost from preflight")
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-train-count", type=int, required=True)
    parser.add_argument("--full-val-count", type=int, required=True)
    parser.add_argument("--target-epochs", type=int, required=True)
    parser.add_argument("--price-per-gpu-hour", type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    completion = _load_json_object(args.completion, error_type=ValueError)
    estimate = estimate_gpu_hours(
        completion,
        full_train_count=args.full_train_count,
        full_val_count=args.full_val_count,
        target_epochs=args.target_epochs,
    )
    payload = estimate.to_dict()
    if args.price_per_gpu_hour is not None:
        payload["cost"] = apply_hourly_price(
            estimate,
            price_per_gpu_hour=args.price_per_gpu_hour,
        )
    _write_json_atomic_no_clobber(args.output, payload)
    print(
        json.dumps(
            str(args.output),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
