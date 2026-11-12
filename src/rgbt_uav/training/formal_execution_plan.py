"""Build a dependency-aware, non-executing formal training plan."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class FormalExecutionPlanError(ValueError):
    """Raised when the formal execution plan is incomplete or unsafe."""


@dataclass(frozen=True)
class FormalExecutionRun:
    """One formal run and the artifacts needed by downstream runs."""

    experiment: str
    seed: int
    config: Path
    output_root: Path
    phase: int
    training_entrypoint: str
    selection_entrypoint: str | None
    dependencies: tuple[Path, ...]
    requires_map_selection: bool
    training_argv: tuple[str, ...]
    selection_argv: tuple[str, ...] | None
    ready: bool
    blockers: tuple[str, ...]


@dataclass(frozen=True)
class FormalExecutionPlan:
    """Validated formal matrix translated into execution phases."""

    protocol_id: str
    matrix: Path
    resource_budget: Path
    authorized: bool
    runs: tuple[FormalExecutionRun, ...]

    def to_metadata(self) -> dict[str, object]:
        """Return a JSON-serializable plan without executing training."""

        return {
            "format_version": 1,
            "status": "planned",
            "execute": False,
            "authorized": self.authorized,
            "protocol_id": self.protocol_id,
            "matrix": _portable(self.matrix),
            "resource_budget": _portable(self.resource_budget),
            "test_evaluated": False,
            "phases": [
                {
                    "phase": phase,
                    "runs": [
                        _run_metadata(run)
                        for run in self.runs
                        if run.phase == phase
                    ],
                }
                for phase in sorted({run.phase for run in self.runs})
            ],
        }


_EXPERIMENTS = {
    "rgb_only": (1, "rgbt-train-formal-unimodal", None, False),
    "ir_only": (1, "rgbt-train-formal-unimodal", None, False),
    "early_fusion_teacher": (
        1,
        "rgbt-train-early-fusion-preflight",
        "rgbt-evaluate-formal-teacher-history",
        True,
    ),
    "gated_teacher": (
        1,
        "rgbt-train-official-gated-teacher",
        "rgbt-evaluate-formal-teacher-history",
        True,
    ),
    "lightweight_detection_only": (
        2,
        "rgbt-train-official-rgb-distillation-preflight",
        "rgbt-evaluate-official-rgb-history",
        True,
    ),
    "lightweight_early_fusion_distillation": (
        2,
        "rgbt-train-official-rgb-distillation-preflight",
        "rgbt-evaluate-official-rgb-history",
        True,
    ),
    "lightweight_gated_distillation": (
        2,
        "rgbt-train-official-rgb-distillation-preflight",
        "rgbt-evaluate-official-rgb-history",
        True,
    ),
}

_CLI_MODULES = {
    "rgbt-train-formal-unimodal": "rgbt_uav.cli.train_formal_unimodal",
    "rgbt-train-early-fusion-preflight": (
        "rgbt_uav.cli.train_early_fusion_preflight"
    ),
    "rgbt-train-official-gated-teacher": (
        "rgbt_uav.cli.train_official_gated_teacher"
    ),
    "rgbt-train-official-rgb-distillation-preflight": (
        "rgbt_uav.cli.train_official_rgb_distillation_preflight"
    ),
    "rgbt-evaluate-formal-teacher-history": (
        "rgbt_uav.cli.evaluate_formal_teacher_history"
    ),
    "rgbt-evaluate-official-rgb-history": (
        "rgbt_uav.cli.evaluate_official_rgb_history"
    ),
}


def _portable(value: Path | str) -> str:
    return str(value).replace("\\", "/")


def _powershell_command(argv: tuple[str, ...]) -> str:
    quoted = [f"'{argument.replace(chr(39), chr(39) * 2)}'" for argument in argv]
    quoted[0] = quoted[0].replace(".venv/python.exe", ".\\.venv\\python.exe")
    return "& " + " ".join(quoted)


def _load_mapping(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FormalExecutionPlanError(
            f"{label} must be a regular non-symlink file: {path}"
        )
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise FormalExecutionPlanError(f"failed to load {label}") from exc
    if not isinstance(payload, dict):
        raise FormalExecutionPlanError(f"{label} root must be a mapping")
    return payload


def _run_metadata(run: FormalExecutionRun) -> dict[str, object]:
    return {
        "experiment": run.experiment,
        "seed": run.seed,
        "config": _portable(run.config),
        "output_root": _portable(run.output_root),
        "training_entrypoint": run.training_entrypoint,
        "selection_entrypoint": run.selection_entrypoint,
        "dependencies": [_portable(path) for path in run.dependencies],
        "requires_map_selection": run.requires_map_selection,
        "training_argv": list(run.training_argv),
        "training_powershell": _powershell_command(run.training_argv),
        "selection_argv": (
            list(run.selection_argv)
            if run.selection_argv is not None
            else None
        ),
        "selection_powershell": (
            _powershell_command(run.selection_argv)
            if run.selection_argv is not None
            else None
        ),
        "ready": run.ready,
        "blockers": list(run.blockers),
        "selection_split": "val",
        "test_evaluated": False,
    }


def _training_argv(entrypoint: str, config: Path) -> tuple[str, ...]:
    argv = (
        ".venv/python.exe",
        "-m",
        _CLI_MODULES[entrypoint],
        "--config",
        _portable(config),
    )
    if entrypoint == "rgbt-train-formal-unimodal":
        return (*argv, "--execute")
    return argv


def _selection_argv(
    *,
    experiment: str,
    entrypoint: str | None,
    output_root: Path,
    data_root: Path,
    device: str,
) -> tuple[str, ...] | None:
    if entrypoint is None:
        return None
    argv = (
        ".venv/python.exe",
        "-m",
        _CLI_MODULES[entrypoint],
        "--run-root",
        _portable(output_root),
    )
    if entrypoint == "rgbt-evaluate-formal-teacher-history":
        kind = "early_fusion" if experiment == "early_fusion_teacher" else "gated"
        argv = (*argv, "--kind", kind)
    return (
        *argv,
        "--data-root",
        _portable(data_root),
        "--device",
        device,
    )


def _config_run(
    raw: Mapping[str, Any],
    *,
    index: int,
    authorized: bool,
) -> FormalExecutionRun:
    experiment = raw.get("experiment")
    seed = raw.get("seed")
    config_value = raw.get("config")
    if experiment not in _EXPERIMENTS:
        raise FormalExecutionPlanError(
            f"matrix run {index} experiment is unsupported"
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed <= 0:
        raise FormalExecutionPlanError(f"matrix run {index} seed is invalid")
    if not isinstance(config_value, str) or not config_value:
        raise FormalExecutionPlanError(f"matrix run {index} config is invalid")

    config = Path(config_value)
    payload = _load_mapping(config, label=f"matrix run {index} config")
    if payload.get("seed") != seed:
        raise FormalExecutionPlanError(f"matrix run {index} seed mismatch")
    output_value = payload.get("output_root")
    if not isinstance(output_value, str) or not output_value:
        raise FormalExecutionPlanError(
            f"matrix run {index} output_root is invalid"
        )
    output_root = Path(output_value)
    if not _portable(output_root).startswith("runs/formal/"):
        raise FormalExecutionPlanError(
            f"matrix run {index} output_root must be under runs/formal"
        )
    data_root_value = payload.get("data_root")
    if (
        experiment not in {"rgb_only", "ir_only"}
        and (not isinstance(data_root_value, str) or not data_root_value)
    ):
        raise FormalExecutionPlanError(
            f"matrix run {index} data_root is invalid"
        )
    device_value = payload.get("device")
    if not isinstance(device_value, str) or not device_value:
        raise FormalExecutionPlanError(
            f"matrix run {index} device is invalid"
        )

    dependencies: tuple[Path, ...] = ()
    teacher_value = payload.get("teacher_checkpoint")
    if experiment.startswith("lightweight_"):
        if not isinstance(teacher_value, str) or not teacher_value:
            raise FormalExecutionPlanError(
                f"matrix run {index} teacher_checkpoint is invalid"
            )
        dependencies = (Path(teacher_value),)

    phase, entrypoint, selection_entrypoint, requires_map_selection = (
        _EXPERIMENTS[experiment]
    )
    blockers: list[str] = []
    if not authorized:
        blockers.append("full_training_not_authorized")
    if output_root.exists() or output_root.is_symlink():
        blockers.append("output_root_exists")
    for dependency in dependencies:
        if not dependency.is_file() or dependency.is_symlink():
            blockers.append(f"missing_dependency:{_portable(dependency)}")
    return FormalExecutionRun(
        experiment=experiment,
        seed=seed,
        config=config,
        output_root=output_root,
        phase=phase,
        training_entrypoint=entrypoint,
        selection_entrypoint=selection_entrypoint,
        dependencies=dependencies,
        requires_map_selection=requires_map_selection,
        training_argv=_training_argv(entrypoint, config),
        selection_argv=_selection_argv(
            experiment=experiment,
            entrypoint=selection_entrypoint,
            output_root=output_root,
            data_root=Path(data_root_value or "."),
            device=device_value,
        ),
        ready=not blockers,
        blockers=tuple(blockers),
    )


def load_formal_execution_plan(
    matrix_path: Path,
    resource_budget_path: Path,
    *,
    require_unauthorized: bool = True,
) -> FormalExecutionPlan:
    """Load the formal matrix and prove that execution is still gated."""

    matrix_path = Path(matrix_path)
    resource_budget_path = Path(resource_budget_path)
    matrix = _load_mapping(matrix_path, label="formal matrix")
    budget = _load_mapping(resource_budget_path, label="resource budget")
    if matrix.get("format_version") != 1:
        raise FormalExecutionPlanError("matrix format_version must be 1")
    protocol_id = matrix.get("protocol_id")
    if not isinstance(protocol_id, str) or not protocol_id:
        raise FormalExecutionPlanError("matrix protocol_id is invalid")
    decision = budget.get("decision")
    if not isinstance(decision, Mapping):
        raise FormalExecutionPlanError("resource budget decision must be a mapping")
    authorized = decision.get("full_training_authorized")
    if not isinstance(authorized, bool):
        raise FormalExecutionPlanError(
            "resource budget authorization must be boolean"
        )
    if require_unauthorized and authorized:
        raise FormalExecutionPlanError(
            "resource budget must remain explicitly unauthorized while planning"
        )
    raw_runs = matrix.get("runs")
    if not isinstance(raw_runs, list) or len(raw_runs) != 21:
        raise FormalExecutionPlanError("formal matrix must contain 21 runs")
    runs = tuple(
        _config_run(raw, index=index, authorized=authorized)
        for index, raw in enumerate(raw_runs)
        if isinstance(raw, Mapping)
    )
    if len(runs) != len(raw_runs):
        raise FormalExecutionPlanError("every matrix run must be a mapping")
    if len({(run.experiment, run.seed) for run in runs}) != len(runs):
        raise FormalExecutionPlanError("formal runs must be unique")

    produced = {
        run.output_root / "best_map50_95.pt"
        for run in runs
        if run.experiment in {"early_fusion_teacher", "gated_teacher"}
    }
    required = {path for run in runs for path in run.dependencies}
    if required != produced:
        raise FormalExecutionPlanError(
            "student teacher dependencies do not match planned teacher outputs"
        )

    return FormalExecutionPlan(
        protocol_id=protocol_id,
        matrix=matrix_path,
        resource_budget=resource_budget_path,
        authorized=authorized,
        runs=runs,
    )
