"""Complete production workload registry for formal calibration."""

from __future__ import annotations

from rgbt_uav.training.formal_calibration_student_workloads import (
    lightweight_student_calibration_adapters,
)
from rgbt_uav.training.formal_calibration_teacher_workloads import (
    paired_teacher_calibration_adapters,
)
from rgbt_uav.training.formal_calibration_unimodal_workloads import (
    unimodal_calibration_adapters,
)
from rgbt_uav.training.formal_calibration_workloads import (
    WorkloadFactory,
    build_formal_workload_factory,
)


def build_production_calibration_workload_factory() -> WorkloadFactory:
    """Build strict dispatch containing all seven production adapters."""

    adapters = {
        **unimodal_calibration_adapters(),
        **paired_teacher_calibration_adapters(),
        **lightweight_student_calibration_adapters(),
    }
    return build_formal_workload_factory(adapters)
