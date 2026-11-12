from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rgbt_uav.data.paired import (
    PairedYoloDataset,
    collate_paired_batch,
)
from rgbt_uav.evaluation.distillation import (
    decode_student_predictions,
)
from rgbt_uav.evaluation.paired import evaluate_paired_coco
from rgbt_uav.models.detectors import TeacherRGBTDetector
from rgbt_uav.training.teacher_checkpoint import (
    load_teacher_training_checkpoint,
)

ROOT = Path("data/processed/yolo_smoke_5pct")
CHECKPOINT = Path(
    "runs/paired/teacher_ohem_v4/teacher-best.pt"
)
OUTPUT = Path(
    "runs/paired/teacher_preflight/paired_coco_metrics.json"
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

dataset = PairedYoloDataset(ROOT, split="val")
loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_paired_batch,
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
    CHECKPOINT,
    model=model,
    optimizer=optimizer,
    map_location=device,
)

model.eval()
predictions = []

with torch.no_grad():
    for batch in loader:
        rgb = batch["rgb"].to(device)
        ir = batch["ir"].to(device)

        output = model(rgb, ir)
        decoded = decode_student_predictions(
            output["predictions"],
            image_size=(
                int(rgb.shape[2]),
                int(rgb.shape[3]),
            ),
            num_classes=5,
            reg_max=16,
            score_threshold=0.001,
            iou_threshold=0.7,
            max_det=300,
        )

        for pair_id, item in zip(
            batch["pair_ids"],
            decoded,
            strict=True,
        ):
            predictions.append(
                {
                    "pair_id": pair_id,
                    "boxes": [
                        [
                            float(value)
                            for value in box
                        ]
                        for box in item["boxes"].cpu().tolist()
                    ],
                    "scores": [
                        float(value)
                        for value in item["scores"].cpu().tolist()
                    ],
                    "labels": [
                        int(value)
                        for value in item["labels"].cpu().tolist()
                    ],
                }
            )

label_to_category_id = {
    0: 1,
    1: 2,
    2: 3,
    3: 4,
    4: 5,
}

category_names = {
    1: "car",
    2: "truck",
    3: "bus",
    4: "van",
    5: "freight_car",
}

metrics = evaluate_paired_coco(
    dataset,
    predictions,
    label_to_category_id=label_to_category_id,
    category_names=category_names,
)

payload = {
    "device": str(device),
    "checkpoint": str(CHECKPOINT),
    "restored_epoch": restored["epoch"],
    "images": len(dataset),
    "prediction_boxes": sum(
        len(item["boxes"])
        for item in predictions
    ),
    "mAP50-95": metrics.ap,
    "mAP50": metrics.ap50,
    "AP75": metrics.ap75,
    "AP_small": metrics.ap_small,
    "AR_small": metrics.ar_small,
    "per_class_AP_small": metrics.per_class_ap_small,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(
    json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)

print(json.dumps(payload, indent=2, ensure_ascii=False))
print("teacher_coco_evaluation_ok=True")