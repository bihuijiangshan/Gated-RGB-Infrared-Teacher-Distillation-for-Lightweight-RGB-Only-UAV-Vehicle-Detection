from __future__ import annotations


def teacher_claim_scope(
    *,
    proposed_map50_95: float,
    baseline_map50_95: float,
    proposed_night_map50_95: float,
    baseline_night_map50_95: float,
) -> str:
    """Return the strongest claim allowed by the pre-registered improvement thresholds."""
    overall_gain = proposed_map50_95 - baseline_map50_95
    night_gain = proposed_night_map50_95 - baseline_night_map50_95
    if overall_gain + 1e-12 >= 0.005:
        return "overall"
    if night_gain + 1e-12 >= 0.01:
        return "night_only"
    return "none"


def evaluate_student_tradeoff(
    *,
    teacher_map50_95: float,
    student_map50_95: float,
    teacher_params_m: float,
    student_params_m: float,
    teacher_flops_g: float,
    student_flops_g: float,
) -> dict[str, float | bool]:
    """Evaluate the student against accuracy-retention and efficiency gates."""
    if teacher_map50_95 <= 0 or teacher_params_m <= 0 or teacher_flops_g <= 0:
        raise ValueError("teacher metrics must be positive")
    accuracy_retained = student_map50_95 / teacher_map50_95
    parameter_reduction = 1.0 - (student_params_m / teacher_params_m)
    flops_reduction = 1.0 - (student_flops_g / teacher_flops_g)
    passes = (
        accuracy_retained + 1e-12 >= 0.97
        and parameter_reduction + 1e-12 >= 0.30
        and flops_reduction + 1e-12 >= 0.35
    )
    return {
        "accuracy_retained": accuracy_retained,
        "parameter_reduction": parameter_reduction,
        "flops_reduction": flops_reduction,
        "passes": passes,
    }

