"""Smoke-test gated RGB/IR teacher to lightweight RGB student distillation."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rgbt_uav.data.paired import (
    PairedYoloDataset,
    collate_paired_batch,
)
from rgbt_uav.losses.official_detection_distillation import (
    OfficialTeacherStudentDistillationLoss,
)
from rgbt_uav.models.official_gated_teacher import (
    build_official_gated_teacher,
)
from rgbt_uav.training.official_distillation_student import (
    build_official_rgb_student,
    load_official_rgb_student_width_reduced_pretrained,
)
from rgbt_uav.training.official_gated_teacher_checkpoint import (
    load_official_gated_teacher_weights,
)
from rgbt_uav.training.official_student_loss import (
    OfficialStudentDetectionLoss,
)
from rgbt_uav.training.official_teacher_student_step import (
    run_official_teacher_student_step,
)

DATA_ROOT = Path(
    "data/processed/yolo_smoke_5pct"
)
TEACHER_CHECKPOINT = Path(
    "runs/paired/official_gated_teacher_v1/"
    "best_map50_95.pt"
)
STUDENT_MODEL = (
    "configs/models/"
    "yolov8_rgb_student_tiny.yaml"
)
STUDENT_PRETRAINED = Path("yolov8n.pt")

NUM_CLASSES = 5
REG_MAX = 16


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "this smoke test requires CUDA"
        )

    device = torch.device("cuda:0")
    cuda_index = (
        device.index
        if device.index is not None
        else torch.cuda.current_device()
    )

    for required in (
        DATA_ROOT,
        TEACHER_CHECKPOINT,
        STUDENT_PRETRAINED,
        Path(STUDENT_MODEL),
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    dataset = PairedYoloDataset(
        DATA_ROOT,
        split="train",
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_paired_batch,
        pin_memory=True,
    )
    batch = next(iter(loader))

    teacher = build_official_gated_teacher(
        model_name="yolov8n.yaml",
        num_classes=NUM_CLASSES,
        device=device,
    )
    teacher_metadata = (
        load_official_gated_teacher_weights(
            TEACHER_CHECKPOINT,
            teacher=teacher,
            map_location=device,
        )
    )

    student = build_official_rgb_student(
        model_name=STUDENT_MODEL,
        num_classes=NUM_CLASSES,
        device=device,
    )
    pretrained_metadata = (
        load_official_rgb_student_width_reduced_pretrained(
            student,
            STUDENT_PRETRAINED,
        )
    )

    detection_criterion = (
        OfficialStudentDetectionLoss(
            student
        )
    )
    distillation_criterion = (
        OfficialTeacherStudentDistillationLoss(
            feature_weight=0.0,
            logit_weight=1.0,
            temperature=2.0,
            num_classes=NUM_CLASSES,
            reg_max=REG_MAX,
            classification_distillation_weight=1.0,
            dfl_distillation_weight=0.0,
        )
    )
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=1e-4,
    )

    teacher_before = {
        name: value.detach().clone()
        for name, value in (
            teacher.state_dict().items()
        )
    }

    student_parameter = next(
        student.parameters()
    )
    student_before = (
        student_parameter.detach().clone()
    )

    rgb = batch["rgb"].to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )
    ir = batch["ir"].to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )

    teacher.eval()
    student.eval()

    with torch.no_grad():
        teacher_output = teacher(rgb, ir)
        student_features, student_predictions = (
            student(rgb)
        )

    teacher_features = teacher_output[
        "features"
    ]
    teacher_predictions = teacher_output[
        "predictions"
    ]
    gates = teacher_output["gates"]

    print("device=", device)
    print("dataset_count=", len(dataset))
    print(
        "batch_rgb=",
        tuple(batch["rgb"].shape),
    )
    print(
        "batch_ir=",
        tuple(batch["ir"].shape),
    )
    print(
        "restored_teacher_epoch=",
        teacher_metadata["epoch"],
    )
    print(
        "teacher_experiment_id=",
        teacher_metadata["experiment_id"],
    )
    print(
        "student_model=",
        STUDENT_MODEL,
    )
    print(
        "student_pretrained_loaded_keys=",
        len(
            pretrained_metadata[
                "loaded_keys"
            ]
        ),
    )
    print(
        "student_pretrained_exact_keys=",
        len(
            pretrained_metadata.get(
                "exact_keys",
                (),
            )
        ),
    )
    print(
        "student_pretrained_sliced_keys=",
        len(
            pretrained_metadata.get(
                "sliced_keys",
                (),
            )
        ),
    )
    print(
        "teacher_features=",
        [
            tuple(value.shape)
            for value in teacher_features
        ],
    )
    print(
        "student_features=",
        [
            tuple(value.shape)
            for value in student_features
        ],
    )
    print(
        "teacher_predictions=",
        [
            tuple(value.shape)
            for value in teacher_predictions
        ],
    )
    print(
        "student_predictions=",
        [
            tuple(value.shape)
            for value in student_predictions
        ],
    )
    print(
        "gate_shapes=",
        [
            tuple(value.shape)
            for value in gates
        ],
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(
        cuda_index
    )

    metrics = (
        run_official_teacher_student_step(
            teacher,
            student,
            detection_criterion,
            distillation_criterion,
            optimizer,
            batch,
            device=device,
            num_classes=NUM_CLASSES,
        )
    )

    torch.cuda.synchronize(device)

    peak_memory_mib = (
        torch.cuda.max_memory_allocated(
            cuda_index
        )
        / 1024**2
    )

    teacher_unchanged = all(
        torch.equal(
            value.detach(),
            teacher_before[name],
        )
        for name, value in (
            teacher.state_dict().items()
        )
    )
    student_changed = not torch.equal(
        student_parameter.detach(),
        student_before,
    )
    metrics_finite = all(
        torch.isfinite(
            torch.tensor(value)
        ).item()
        for value in metrics.values()
    )

    teacher_trainable_parameters = sum(
        parameter.numel()
        for parameter in teacher.parameters()
        if parameter.requires_grad
    )
    student_trainable_parameters = sum(
        parameter.numel()
        for parameter in student.parameters()
        if parameter.requires_grad
    )

    gate_sums = [
        gate.sum(
            dim=1
        )
        for gate in gates
    ]
    gates_normalized = all(
        torch.allclose(
            value,
            torch.ones_like(value),
            atol=1e-6,
            rtol=1e-6,
        )
        for value in gate_sums
    )

    print(
        "output_distillation_kind=",
        "detection_aware",
    )
    print(
        "classification_distillation_weight=",
        1.0,
    )
    print(
        "dfl_distillation_weight=",
        0.0,
    )
    print("metrics=", metrics)
    print(
        "teacher_unchanged=",
        teacher_unchanged,
    )
    print(
        "student_changed=",
        student_changed,
    )
    print(
        "metrics_finite=",
        metrics_finite,
    )
    print(
        "gates_normalized=",
        gates_normalized,
    )
    print(
        "teacher_trainable_parameters=",
        teacher_trainable_parameters,
    )
    print(
        "student_trainable_parameters=",
        student_trainable_parameters,
    )
    print(
        "peak_memory_mib=",
        peak_memory_mib,
    )

    assert teacher_metadata["epoch"] == 8
    assert (
        teacher_metadata["experiment_id"]
        == "official-gated-teacher-v1-seed-3407"
    )
    assert len(teacher_features) == 3
    assert len(student_features) == 3
    assert len(teacher_predictions) == 3
    assert len(student_predictions) == 3
    assert len(gates) == 3

    for teacher_prediction in (
        teacher_predictions
    ):
        assert (
            teacher_prediction.shape[1]
            == NUM_CLASSES + 4 * REG_MAX
        )
        assert torch.isfinite(
            teacher_prediction
        ).all()

    for student_prediction in (
        student_predictions
    ):
        assert (
            student_prediction.shape[1]
            == NUM_CLASSES + 4 * REG_MAX
        )
        assert torch.isfinite(
            student_prediction
        ).all()

    assert teacher_unchanged
    assert student_changed
    assert metrics_finite
    assert gates_normalized
    assert teacher_trainable_parameters == 0
    assert student_trainable_parameters > 0

    assert metrics["total"] > 0.0
    assert metrics["detection"] > 0.0
    assert metrics["distillation"] > 0.0
    assert metrics["logit"] > 0.0
    assert metrics["box"] >= 0.0
    assert metrics["classification"] >= 0.0
    assert metrics["dfl"] >= 0.0

    # The reported logit diagnostic contains the raw sum of
    # classification and DFL distillation. With DFL weight set
    # to zero, only classification distillation contributes to
    # the weighted distillation total.
    assert 0.0 < metrics["distillation"] <= metrics["logit"]

    print(
        "official_gated_lightweight_"
        "distillation_step_ok=True"
    )


if __name__ == "__main__":
    main()