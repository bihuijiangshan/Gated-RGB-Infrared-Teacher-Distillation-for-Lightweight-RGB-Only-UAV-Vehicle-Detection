"""Training losses for teacher-student experiments."""

from rgbt_uav.losses.distillation import DetectionDistillationLoss, FeatureDistiller

__all__ = ["DetectionDistillationLoss", "FeatureDistiller"]

