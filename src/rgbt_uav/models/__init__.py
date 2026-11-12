"""PyTorch modules for reliability-aware RGB-IR detection."""

from rgbt_uav.models.detectors import StudentRGBTDetector, TeacherRGBTDetector
from rgbt_uav.models.fusion import ReliabilityGatedFusion

__all__ = ["ReliabilityGatedFusion", "StudentRGBTDetector", "TeacherRGBTDetector"]

