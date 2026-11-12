from pathlib import Path

from rgbt_uav.training.distillation_config import DistillationConfig
from rgbt_uav.training.teacher_runner import run_teacher_preflight

config = DistillationConfig.from_yaml(
    Path(
        "configs/experiments/"
        "preflight_teacher_ohem_v4.yaml"
    )
)

result = run_teacher_preflight(config)

print("best_weights=", result.best_weights)
print("last_weights=", result.last_weights)
print("completed_epochs=", result.completed_epochs)
print("metrics=", result.metrics)
print("teacher_preflight_ok=True")