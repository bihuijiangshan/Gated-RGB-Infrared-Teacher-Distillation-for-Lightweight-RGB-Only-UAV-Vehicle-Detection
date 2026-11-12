from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rgbt_uav.data.paired import (
    PairedYoloDataset,
    collate_paired_batch,
)
from rgbt_uav.losses.detection import PairedDetectionLoss
from rgbt_uav.models.detectors import TeacherRGBTDetector
from rgbt_uav.training.teacher_checkpoint import (
    load_teacher_training_checkpoint,
)

root = Path("data/processed/yolo_smoke_5pct")
checkpoint = Path(
    "runs/paired/teacher_balanced_v2/teacher-best.pt"
)

dataset = PairedYoloDataset(root, split="val")
loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_paired_batch,
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = TeacherRGBTDetector(
    num_classes=5,
    base_channels=32,
    reg_max=16,
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
)

restored = load_teacher_training_checkpoint(
    checkpoint,
    model=model,
    optimizer=optimizer,
    map_location=device,
)

criterion = PairedDetectionLoss(
    num_classes=5,
    reg_max=16,
)

batch = next(iter(loader))
rgb = batch["rgb"].to(device)
ir = batch["ir"].to(device)
boxes = [value.to(device) for value in batch["boxes"]]
labels = [value.to(device) for value in batch["labels"]]


def inspect_mode(training: bool) -> None:
    model.train(training)

    with torch.no_grad():
        output = model(rgb, ir)
        metrics = criterion(
            output["predictions"],
            boxes,
            labels,
        )

    print("training=", training)
    print("total=", float(metrics["total"]))
    print("classification=", float(metrics["classification"]))
    print("box=", float(metrics["box"]))

    for index, level in enumerate(output["predictions"]):
        classes = level[:, :5].sigmoid()
        regression = level[:, 5:9].sigmoid()

        print(
            "level=",
            index,
            "score_min=",
            float(classes.min()),
            "score_max=",
            float(classes.max()),
            "score_mean=",
            float(classes.mean()),
            "box_min=",
            float(regression.min()),
            "box_max=",
            float(regression.max()),
            "box_mean=",
            float(regression.mean()),
        )


print("device=", device)
print("restored_epoch=", restored["epoch"])
inspect_mode(training=True)
inspect_mode(training=False)
print("teacher_train_eval_diagnosis_ok=True")