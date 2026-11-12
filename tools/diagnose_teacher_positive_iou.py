from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rgbt_uav.data.paired import (
    PairedYoloDataset,
    collate_paired_batch,
)
from rgbt_uav.models.detectors import TeacherRGBTDetector
from rgbt_uav.training.teacher_checkpoint import (
    load_teacher_training_checkpoint,
)


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center_x = boxes[:, 0]
    center_y = boxes[:, 1]
    width = boxes[:, 2]
    height = boxes[:, 3]

    return torch.stack(
        (
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2,
        ),
        dim=1,
    )


def aligned_iou(
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    intersection_min = torch.maximum(
        first[:, :2],
        second[:, :2],
    )
    intersection_max = torch.minimum(
        first[:, 2:],
        second[:, 2:],
    )
    intersection_size = (
        intersection_max - intersection_min
    ).clamp(min=0)

    intersection = (
        intersection_size[:, 0]
        * intersection_size[:, 1]
    )

    first_size = (
        first[:, 2:] - first[:, :2]
    ).clamp(min=0)
    second_size = (
        second[:, 2:] - second[:, :2]
    ).clamp(min=0)

    first_area = first_size[:, 0] * first_size[:, 1]
    second_area = second_size[:, 0] * second_size[:, 1]

    union = first_area + second_area - intersection

    return intersection / union.clamp(min=1e-9)


root = Path("data/processed/yolo_smoke_5pct")
checkpoint = Path(
    "runs/paired/teacher_ohem_v4/teacher-best.pt"
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

batch = next(iter(loader))
rgb = batch["rgb"].to(device)
ir = batch["ir"].to(device)

model.eval()

with torch.no_grad():
    output = model(rgb, ir)

sample_boxes = batch["boxes"][0].to(device)
sample_labels = batch["labels"][0].to(device)
target_xyxy = xywh_to_xyxy(sample_boxes)

print("device=", device)
print("restored_epoch=", restored["epoch"])
print("pair_id=", batch["pair_ids"][0])
print("targets=", len(sample_boxes))

for level_index, prediction in enumerate(
    output["predictions"]
):
    _, _, height, width = prediction.shape

    predicted_boxes = []
    positive_scores = []
    occupied_cells = set()

    for box, label in zip(
        sample_boxes,
        sample_labels,
        strict=True,
    ):
        cell_x = min(
            width - 1,
            max(0, int(float(box[0]) * width)),
        )
        cell_y = min(
            height - 1,
            max(0, int(float(box[1]) * height)),
        )

        occupied_cells.add((cell_x, cell_y))

        raw_box = prediction[
            0,
            5:9,
            cell_y,
            cell_x,
        ]

        decoded_box = raw_box.sigmoid()
        decoded_box = decoded_box.clone()

        decoded_box[0] = (cell_x + decoded_box[0]) / width
        decoded_box[1] = (cell_y + decoded_box[1]) / height

        predicted_boxes.append(decoded_box)

        positive_scores.append(
            prediction[
                0,
                int(label),
                cell_y,
                cell_x,
            ].sigmoid()
        )

    predicted_xywh = torch.stack(predicted_boxes)
    predicted_xyxy = xywh_to_xyxy(
        predicted_xywh
    )
    ious = aligned_iou(
        predicted_xyxy,
        target_xyxy,
    )
    scores = torch.stack(positive_scores)

    print("level=", level_index)
    print("grid=", (height, width))
    print("unique_positive_cells=", len(occupied_cells))
    print(
        "collisions=",
        len(sample_boxes) - len(occupied_cells),
    )
    print("positive_score_mean=", float(scores.mean()))
    print("positive_score_min=", float(scores.min()))
    print("positive_score_max=", float(scores.max()))
    print("positive_iou_mean=", float(ious.mean()))
    print("positive_iou_max=", float(ious.max()))
    print(
        "positive_iou_ge_050=",
        int((ious >= 0.5).sum()),
    )
    print(
        "predicted_wh_mean=",
        predicted_xywh[:, 2:].mean(dim=0).tolist(),
    )
    print(
        "target_wh_mean=",
        sample_boxes[:, 2:].mean(dim=0).tolist(),
    )

print("teacher_positive_iou_diagnosis_ok=True")