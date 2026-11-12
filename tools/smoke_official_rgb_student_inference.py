from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rgbt_uav.data.paired import (
    PairedYoloDataset,
    collate_paired_batch,
)
from rgbt_uav.evaluation.official_rgb_student import (
    run_official_rgb_student_inference,
)
from rgbt_uav.training.official_distillation_student import (
    build_official_rgb_student,
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

data_root = Path("data/processed/yolo_smoke_5pct")
checkpoint_path = Path(
    "runs/paired/official_rgb_distillation_v1/best.pt"
)

if not checkpoint_path.is_file():
    raise FileNotFoundError(checkpoint_path)

dataset = PairedYoloDataset(
    data_root,
    split="val",
)
loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_paired_batch,
)

student = build_official_rgb_student(
    model_name="yolov8n.yaml",
    num_classes=5,
    device=device,
)

payload = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False,
)
probe_batch = next(iter(loader))
print(
    "batch_keys=",
    sorted(probe_batch),  # noqa: F821
)
print(
    "pair_ids_container_type=",
    type(probe_batch["pair_ids"]).__name__,
)
print(
    "pair_ids_container=",
    probe_batch["pair_ids"],
)

if not isinstance(payload, dict):
    raise RuntimeError(
        "student checkpoint must be a dictionary"
    )
if "student" not in payload:
    raise RuntimeError(
        "student checkpoint is missing student state"
    )

student.load_state_dict(
    payload["student"],
    strict=True,
)
restored_epoch = int(payload.get("epoch", -1))

results = run_official_rgb_student_inference(
    student,
    loader,
    device=device,
    score_threshold=0.001,
    iou_threshold=0.7,
    max_det=300,
)

prediction_boxes = sum(
    len(result["boxes"])
    for result in results
)
scores = [
    score
    for result in results
    for score in result["scores"]
]
labels = [
    label
    for result in results
    for label in result["labels"]
]

print("device=", device)
print("checkpoint=", checkpoint_path)
print("restored_epoch=", restored_epoch)
print("dataset_count=", len(dataset))
print("prediction_images=", len(results))
print("prediction_boxes=", prediction_boxes)

if scores:
    print("score_min=", min(scores))
    print("score_max=", max(scores))
    print(
        "score_mean=",
        sum(scores) / len(scores),
    )
else:
    print("score_min=", None)
    print("score_max=", None)
    print("score_mean=", None)

print(
    "label_counts=",
    {
        label: labels.count(label)
        for label in sorted(set(labels))
    },
)

assert restored_epoch == 1
assert len(results) == len(dataset)
assert all(
    result["pair_id"] == dataset[index]["pair_id"]
    for index, result in enumerate(results)
)
assert all(
    0 <= label < 5
    for label in labels
)
assert all(
    0.0 <= score <= 1.0
    for score in scores
)

print("official_rgb_student_inference_ok=True")