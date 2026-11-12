"""No-clobber retention of formal per-epoch checkpoints."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any


class FormalEpochCheckpointError(
    RuntimeError
):
    """Raised when an epoch checkpoint cannot be retained safely."""


class FormalEpochCheckpointRetainer:
    """Copy Ultralytics last.pt into normalized epoch history."""

    def __init__(
        self,
        destination: Path,
        *,
        expected_epochs: int,
        starting_epoch: int = 0,
    ) -> None:
        if (
            isinstance(
                expected_epochs,
                bool,
            )
            or not isinstance(
                expected_epochs,
                int,
            )
            or expected_epochs <= 0
        ):
            raise FormalEpochCheckpointError(
                "expected_epochs must be a positive integer"
            )

        if (
            isinstance(starting_epoch, bool)
            or not isinstance(starting_epoch, int)
            or starting_epoch < 0
            or starting_epoch >= expected_epochs
        ):
            raise FormalEpochCheckpointError(
                "starting_epoch must be a non-negative integer "
                "smaller than expected_epochs"
            )

        self.destination = Path(
            destination
        )
        self.expected_epochs = (
            expected_epochs
        )
        self._retained_epochs = list(
            range(1, starting_epoch + 1)
        )
        self._retained_paths = [
            self.destination
            / f"epoch-{epoch:03d}.pt"
            for epoch in self._retained_epochs
        ]

        invalid_existing = [
            path
            for path in self._retained_paths
            if (
                not path.is_file()
                or path.is_symlink()
            )
        ]
        if invalid_existing:
            raise FormalEpochCheckpointError(
                "existing checkpoint history is incomplete: "
                f"{invalid_existing[0]}"
            )

    @property
    def retained_epochs(
        self,
    ) -> tuple[int, ...]:
        """Return retained one-based epochs."""

        return tuple(
            self._retained_epochs
        )

    @property
    def retained_paths(
        self,
    ) -> tuple[Path, ...]:
        """Return retained checkpoint paths."""

        return tuple(
            self._retained_paths
        )

    def __call__(
        self,
        trainer: Any,
    ) -> Path:
        """Retain the checkpoint written immediately before the callback."""

        raw_epoch = getattr(
            trainer,
            "epoch",
            None,
        )

        if (
            isinstance(raw_epoch, bool)
            or not isinstance(
                raw_epoch,
                int,
            )
            or raw_epoch < 0
            or raw_epoch
            >= self.expected_epochs
        ):
            raise FormalEpochCheckpointError(
                "trainer epoch must be a valid "
                "zero-based epoch index"
            )

        epoch = raw_epoch + 1
        destination = (
            self.destination
            / f"epoch-{epoch:03d}.pt"
        )

        if (
            destination.exists()
            or destination.is_symlink()
        ):
            raise FormalEpochCheckpointError(
                "epoch checkpoint already exists: "
                f"{destination}"
            )

        expected_epoch = len(
            self._retained_epochs
        ) + 1
        if epoch != expected_epoch:
            raise FormalEpochCheckpointError(
                "trainer epoch is not the next sequential epoch: "
                f"expected {expected_epoch}, received {epoch}"
            )
        source_value = getattr(
            trainer,
            "last",
            None,
        )

        if not isinstance(
            source_value,
            (str, Path),
        ):
            raise FormalEpochCheckpointError(
                "trainer last checkpoint path is invalid"
            )

        source = Path(source_value)

        if (
            not source.is_file()
            or source.is_symlink()
        ):
            raise FormalEpochCheckpointError(
                "last checkpoint must be a regular "
                f"non-symlink file: {source}"
            )

        try:
            self.destination.mkdir(
                parents=True,
                exist_ok=True,
            )
        except OSError as exc:
            raise FormalEpochCheckpointError(
                "failed to create epoch checkpoint directory: "
                f"{self.destination}"
            ) from exc

        destination_created = False

        try:
            with source.open(
                "rb"
            ) as source_handle:
                with destination.open(
                    "xb"
                ) as destination_handle:
                    destination_created = True
                    shutil.copyfileobj(
                        source_handle,
                        destination_handle,
                        length=1024 * 1024,
                    )
                    destination_handle.flush()
        except FileExistsError as exc:
            raise FormalEpochCheckpointError(
                "epoch checkpoint already exists: "
                f"{destination}"
            ) from exc
        except OSError as exc:
            if destination_created:
                try:
                    destination.unlink(
                        missing_ok=True
                    )
                except OSError:
                    pass

            raise FormalEpochCheckpointError(
                "failed to retain epoch checkpoint: "
                f"{destination}"
            ) from exc

        self._retained_epochs.append(
            epoch
        )
        self._retained_paths.append(
            destination
        )

        return destination
