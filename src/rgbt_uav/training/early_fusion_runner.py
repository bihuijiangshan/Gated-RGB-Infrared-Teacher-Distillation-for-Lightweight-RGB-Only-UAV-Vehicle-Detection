from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from ultralytics.utils.loss import v8DetectionLoss

from rgbt_uav.data.paired import PairedYoloDataset, collate_paired_batch
from rgbt_uav.training.checkpoint import (
    load_early_fusion_checkpoint,
    save_early_fusion_checkpoint,
)
from rgbt_uav.training.early_fusion import train_early_fusion_epoch
from rgbt_uav.training.early_fusion_config import EarlyFusionConfig
from rgbt_uav.training.early_fusion_pretrained import load_adapted_state_dict
from rgbt_uav.training.early_fusion_runtime import (
    build_early_fusion_model,
    evaluate_early_fusion_loss,
    resolve_early_fusion_device,
    seed_early_fusion,
)
from rgbt_uav.training.early_fusion_state import EarlyFusionState
from rgbt_uav.training.ultralytics_adapter import (
    run_ultralytics_early_fusion_step,
)


class EarlyFusionRunnerError(RuntimeError):
    """Raised when an early-fusion run cannot be completed safely."""


@dataclass(frozen=True)
class EarlyFusionRunResult:
    """Published artifacts and metrics from an early-fusion run."""

    best_weights: Path
    last_weights: Path
    state_path: Path
    completion_path: Path
    metrics: dict[str, float]
    completed_epochs: tuple[int, ...]
    epoch_weights: tuple[Path, ...]


DatasetFactory = Callable[[Path, str], Any]


def _existing_epoch_weights(
    output_root: Path,
    *,
    completed_epoch: int,
) -> list[Path]:
    paths = [
        output_root / "epochs" / f"epoch-{epoch:03d}.pt"
        for epoch in range(1, completed_epoch + 1)
    ]
    invalid = [
        path
        for path in paths
        if not path.is_file() or path.is_symlink()
    ]
    if invalid:
        raise EarlyFusionRunnerError(
            "resume epoch checkpoint history is incomplete: "
            f"{invalid[0]}"
        )
    return paths


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        temporary = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        )
        temporary_path = Path(temporary.name)

        with temporary:
            json.dump(
                payload,
                temporary,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
                indent=2,
            )
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())

        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def _make_loader(
    dataset: Any,
    *,
    batch: int,
    shuffle: bool,
    workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_paired_batch,
    )


def _metrics_for_checkpoint(
    train_metrics: dict[str, float],
    val_loss: float,
) -> dict[str, float]:
    metrics = dict(train_metrics)
    metrics["val_loss"] = float(val_loss)
    return metrics


def run_early_fusion_preflight(
    config: EarlyFusionConfig,
    *,
    dataset_factory: DatasetFactory | None = None,
) -> EarlyFusionRunResult:
    """Run or resume the configured early-fusion preflight."""

    if not isinstance(config, EarlyFusionConfig):
        raise EarlyFusionRunnerError(
            "config must be an EarlyFusionConfig"
        )

    if dataset_factory is None:
        dataset_factory = PairedYoloDataset

    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    last_weights = output_root / "last.pt"
    best_weights = output_root / "best.pt"
    state_path = output_root / "run_state.json"
    completion_path = output_root / "completion.json"

    if state_path.exists() and not config.resume:
        raise EarlyFusionRunnerError(
            "run state already exists; set resume=true to continue"
        )
    
    try:
        device = resolve_early_fusion_device(config.device)
        seed_early_fusion(config.seed)

        train_dataset = dataset_factory(
            Path(config.data_root),
            split="train",
        )
        val_dataset = dataset_factory(
            Path(config.data_root),
            split="val",
        )

        train_loader = _make_loader(
            train_dataset,
            batch=config.batch,
            shuffle=True,
            workers=config.workers,
        )
        val_loader = _make_loader(
            val_dataset,
            batch=config.batch,
            shuffle=False,
            workers=config.workers,
        )

        model = build_early_fusion_model(
            model_name=config.model,
            num_classes=config.num_classes,
            device=device,
        )

        if config.pretrained is not None and not config.resume:
            pretrained_state = _load_pretrained_state_dict(config.pretrained)
            load_adapted_state_dict(
                model,
                pretrained_state,
                strict=False,
            )

        criterion = v8DetectionLoss(model)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
        )

        completed_epochs: list[int] = []
        epoch_weights: list[Path] = []
        best_val_loss: float | None = None
        start_epoch = 1

        if config.resume:
            if not state_path.is_file():
                raise EarlyFusionRunnerError(
                    f"resume state does not exist: {state_path}"
                )

            state = EarlyFusionState.from_json(state_path)

            if state.experiment_id != config.experiment_id:
                raise EarlyFusionRunnerError(
                    "resume state experiment_id does not match configuration"
                )

            if not last_weights.is_file():
                raise EarlyFusionRunnerError(
                    f"resume checkpoint does not exist: {last_weights}"
                )

            restored = load_early_fusion_checkpoint(
                last_weights,
                model=model,
                optimizer=optimizer,
                map_location=device,
            )

            restored_epoch = restored["epoch"]
            if not isinstance(restored_epoch, int):
                raise EarlyFusionRunnerError(
                    "checkpoint epoch must be an integer"
                )

            if restored_epoch != state.completed_epoch:
                raise EarlyFusionRunnerError(
                    "checkpoint epoch does not match run state"
                )

            start_epoch = state.completed_epoch + 1
            best_val_loss = state.best_val_loss
            epoch_weights.extend(
                _existing_epoch_weights(
                    output_root,
                    completed_epoch=state.completed_epoch,
                )
            )

            if state.best_checkpoint is not None:
                best_weights = Path(state.best_checkpoint)


        for epoch in range(start_epoch, config.epochs + 1):
            train_metrics = train_early_fusion_epoch(
                model,
                criterion,
                optimizer,
                train_loader,
                device=device,
                step_fn=run_ultralytics_early_fusion_step,
            )

            val_loss = evaluate_early_fusion_loss(
                model,
                criterion,
                val_loader,
                device=device,
            )

            metrics = _metrics_for_checkpoint(
                train_metrics,
                val_loss,
            )

            save_early_fusion_checkpoint(
                last_weights,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
            )

            epoch_checkpoint = (
                output_root / "epochs" / f"epoch-{epoch:03d}.pt"
            )
            if epoch_checkpoint.exists() or epoch_checkpoint.is_symlink():
                raise EarlyFusionRunnerError(
                    "epoch checkpoint already exists: "
                    f"{epoch_checkpoint}"
                )
            save_early_fusion_checkpoint(
                epoch_checkpoint,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
            )
            epoch_weights.append(epoch_checkpoint)

            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = float(val_loss)
                save_early_fusion_checkpoint(
                    best_weights,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=metrics,
                )

            completed_epochs.append(epoch)

            running_state = EarlyFusionState(
                experiment_id=config.experiment_id,
                status="running",
                completed_epoch=epoch,
                best_val_loss=best_val_loss,
                last_checkpoint=last_weights,
                best_checkpoint=best_weights,
            )
            running_state.write_json(state_path)

        if not completed_epochs:
            raise EarlyFusionRunnerError(
                "no epoch was executed"
            )

        final_metrics = {
            "val_loss": float(best_val_loss)
            if best_val_loss is not None
            else float("inf"),
        }

        completed_state = EarlyFusionState(
            experiment_id=config.experiment_id,
            status="completed",
            completed_epoch=completed_epochs[-1],
            best_val_loss=best_val_loss,
            last_checkpoint=last_weights,
            best_checkpoint=best_weights,
        )
        completed_state.write_json(state_path)

        all_completed_epochs = list(
            range(1, completed_epochs[-1] + 1)
        )

        _write_json_atomic(
            completion_path,
            {
                "status": "completed",
                "experiment_id": config.experiment_id,
                "experiment_group": config.experiment_group,
                "config": {
                    "model": config.model,
                    "epochs": config.epochs,
                    "batch": config.batch,
                    "imgsz": config.imgsz,
                    "seed": config.seed,
                    "device": config.device,
                    "workers": config.workers,
                    "lr": config.lr,
                    "num_classes": config.num_classes,
                    "resume": config.resume,
                    "pretrained": (
                        str(config.pretrained) if config.pretrained is not None else None
                    ),
                },
                "dataset": {
                    "train_count": len(train_dataset),
                    "val_count": len(val_dataset),
                },
                "completed_epochs": completed_epochs,
                "all_completed_epochs": all_completed_epochs,
                "best_weights": str(best_weights),
                "last_weights": str(last_weights),
                "epoch_weights": [str(path) for path in epoch_weights],
                "metrics": final_metrics,
            },
        )

        return EarlyFusionRunResult(
            best_weights=best_weights,
            last_weights=last_weights,
            state_path=state_path,
            completion_path=completion_path,
            metrics=final_metrics,
            completed_epochs=tuple(completed_epochs),
            epoch_weights=tuple(epoch_weights),
        )

    except EarlyFusionRunnerError:
        raise
    except Exception as exc:
        raise EarlyFusionRunnerError(
            f"early-fusion run failed: {type(exc).__name__}: {exc}"
        ) from exc

def _load_pretrained_state_dict(path: Path) -> dict[str, torch.Tensor]:
    checkpoint_path = Path(path)

    if not checkpoint_path.is_file():
        raise EarlyFusionRunnerError(
            f"pretrained checkpoint does not exist: {checkpoint_path}"
        )

    try:
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as exc:
        raise EarlyFusionRunnerError(
            f"could not load pretrained checkpoint: {checkpoint_path}"
        ) from exc

    if hasattr(payload, "state_dict"):
        state_dict = payload.state_dict()
    elif isinstance(payload, dict) and "model" in payload:
        model_payload = payload["model"]
        if hasattr(model_payload, "state_dict"):
            state_dict = model_payload.state_dict()
        elif isinstance(model_payload, dict):
            state_dict = model_payload
        else:
            raise EarlyFusionRunnerError(
                "pretrained model payload is unsupported"
            )
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise EarlyFusionRunnerError(
            "pretrained checkpoint payload is unsupported"
        )

    if not isinstance(state_dict, dict):
        raise EarlyFusionRunnerError(
            "pretrained state_dict must be a mapping"
        )

    return state_dict
