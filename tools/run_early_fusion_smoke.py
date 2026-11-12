from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss

from rgbt_uav.data.paired import (
    PairedYoloDataset,
    collate_paired_batch,
)
from rgbt_uav.training.checkpoint import (
    load_early_fusion_checkpoint,
    save_early_fusion_checkpoint,
)
from rgbt_uav.training.early_fusion import train_early_fusion_epoch
from rgbt_uav.training.ultralytics_adapter import (
    run_ultralytics_early_fusion_step,
    to_ultralytics_early_fusion_batch,
)


def build_model(device: torch.device) -> DetectionModel:
    model = DetectionModel(
        "yolov8n.yaml",
        ch=4,
        nc=5,
        verbose=False,
    ).to(device)

    model.args = SimpleNamespace(
        box=7.5,
        cls=0.5,
        dfl=1.5,
    )
    return model


def evaluate_loss(
    model: DetectionModel,
    criterion: v8DetectionLoss,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    batches = 0

    with torch.no_grad():
        for paired_batch in loader:
            converted = to_ultralytics_early_fusion_batch(
                paired_batch
            )

            train_batch = {
                "img": converted["img"].to(device),
                "batch_idx": converted["batch_idx"].to(device),
                "cls": converted["cls"].to(device),
                "bboxes": converted["bboxes"].to(device),
            }

            predictions = model(train_batch["img"])
            loss, _ = criterion(predictions, train_batch)

            total += float(loss.detach().cpu())
            batches += 1

    if batches == 0:
        raise RuntimeError("validation loader produced no batches")

    return total / batches


def main() -> None:
    torch.set_num_threads(1)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    root = Path("data/processed/yolo_smoke_5pct")
    output = Path(
        "runs/paired/early_fusion_official_smoke_20260819"
    )
    checkpoint_path = output / "epoch-001.pt"

    train_dataset = PairedYoloDataset(
        root,
        split="train",
    )
    val_dataset = PairedYoloDataset(
        root,
        split="val",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_paired_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_paired_batch,
    )

    model = build_model(device)
    criterion = v8DetectionLoss(model)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
    )

    train_metrics = train_early_fusion_epoch(
        model,
        criterion,
        optimizer,
        train_loader,
        device=device,
        step_fn=run_ultralytics_early_fusion_step,
    )

    val_loss = evaluate_loss(
        model,
        criterion,
        val_loader,
        device,
    )

    save_early_fusion_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=1,
        metrics={
            **train_metrics,
            "val_loss": val_loss,
        },
    )

    reloaded_model = build_model(device)
    reloaded_criterion = v8DetectionLoss(reloaded_model)
    reloaded_optimizer = torch.optim.Adam(
        reloaded_model.parameters(),
        lr=1e-4,
    )

    restored = load_early_fusion_checkpoint(
        checkpoint_path,
        model=reloaded_model,
        optimizer=reloaded_optimizer,
        map_location=device,
    )

    reloaded_val_loss = evaluate_loss(
        reloaded_model,
        reloaded_criterion,
        val_loader,
        device,
    )

    print("device=", device)
    print("train_count=", len(train_dataset))
    print("val_count=", len(val_dataset))
    print("train_metrics=", train_metrics)
    print("val_loss=", val_loss)
    print("reloaded_val_loss=", reloaded_val_loss)
    print("restored_epoch=", restored["epoch"])
    print("checkpoint_path=", checkpoint_path)
    print("early_fusion_smoke_ok=True")


if __name__ == "__main__":
    main()