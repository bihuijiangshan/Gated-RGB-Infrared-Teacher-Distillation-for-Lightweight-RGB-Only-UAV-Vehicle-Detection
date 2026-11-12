"""Guard execution of one explicitly authorized formal training run."""

from __future__ import annotations

import hashlib
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from rgbt_uav.training.formal_execution_plan import (
    FormalExecutionRun,
    load_formal_execution_plan,
)


class FormalRunExecutionError(RuntimeError):
    """Raised before or during an unsafe formal run invocation."""


@dataclass(frozen=True)
class FormalRunExecutionResult:
    """Evidence that exactly one formal training command completed."""

    experiment: str
    seed: int
    config: Path
    output_root: Path
    git_commit: str
    command: tuple[str, ...]
    return_code: int


CommandRunner = Callable[[Sequence[str]], int]
GitCommitReader = Callable[[], str]
WorktreeChecker = Callable[[], bool]


def _load_mapping(path: Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise FormalRunExecutionError(
            f"document must be a regular non-symlink file: {source}"
        )
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise FormalRunExecutionError("failed to load document") from exc
    if not isinstance(payload, dict):
        raise FormalRunExecutionError("document root must be a mapping")
    return payload


def _git_output(*arguments: str) -> str:
    try:
        result = subprocess.run(
            ("git", *arguments),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FormalRunExecutionError("failed to inspect git state") from exc
    return result.stdout.strip()


def _current_git_commit() -> str:
    return _git_output("rev-parse", "HEAD")


def _tracked_worktree_clean() -> bool:
    return not _git_output("status", "--porcelain", "--untracked-files=no")


def _run_command(argv: Sequence[str]) -> int:
    try:
        return subprocess.run(tuple(argv), check=False).returncode
    except OSError as exc:
        raise FormalRunExecutionError("failed to start training command") from exc


def _authorization_commit(
    budget: Mapping[str, Any],
    *,
    protocol_id: str,
) -> str:
    decision = budget
    if not isinstance(decision, Mapping):
        raise FormalRunExecutionError(
            "resource budget decision must be a mapping"
        )
    if decision.get("format_version") != 1 or decision.get("status") != "approved":
        raise FormalRunExecutionError("formal run is not authorized: approval required")
    for field in ("authorized_by", "authorized_at_utc"):
        value = decision.get(field)
        if not isinstance(value, str) or not value.strip():
            raise FormalRunExecutionError(
                "formal authorization record is incomplete"
            )
    if decision.get("authorized_protocol_id") != protocol_id:
        raise FormalRunExecutionError(
            "authorized protocol does not match the plan"
        )
    try:
        approved_at = datetime.fromisoformat(decision["authorized_at_utc"])
        if approved_at.utcoffset() != timezone.utc.utcoffset(approved_at):
            raise ValueError("UTC required")
    except (TypeError, ValueError) as exc:
        raise FormalRunExecutionError("authorized_at_utc must be a UTC timestamp") from exc
    commit = decision.get("authorized_git_commit")
    if (
        not isinstance(commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", commit) is None
    ):
        raise FormalRunExecutionError("authorized git commit is invalid")
    return commit


def _bound_hashes(matrix: Path, budget: Path, protocol: Path, config: Path) -> dict[str, str]:
    return {
        label: hashlib.sha256(path.read_bytes()).hexdigest()
        for label, path in (
            ("matrix", matrix), ("resource_budget", budget),
            ("protocol", protocol), ("config", config),
        )
    }


def build_authorization_template(
    matrix_path: Path, resource_budget_path: Path, protocol_path: Path,
    *, experiment: str, seed: int,
    git_commit_reader: GitCommitReader = _current_git_commit,
) -> dict[str, Any]:
    """Build a draft for exactly one run; never grant authorization."""
    plan = load_formal_execution_plan(matrix_path, resource_budget_path)
    run = _select_run(plan.runs, experiment=experiment, seed=seed)
    protocol = _load_mapping(protocol_path)
    if protocol.get("protocol_id") != plan.protocol_id:
        raise FormalRunExecutionError("protocol does not match the plan")
    return {
        "format_version": 1, "status": "draft",
        "authorized_by": "", "authorized_at_utc": "",
        "authorized_git_commit": git_commit_reader(),
        "authorized_protocol_id": plan.protocol_id,
        "runs": [{"experiment": experiment, "seed": seed}],
        "sha256": _bound_hashes(matrix_path, resource_budget_path, protocol_path, run.config),
    }


def _select_run(
    runs: Sequence[FormalExecutionRun],
    *,
    experiment: str,
    seed: int,
) -> FormalExecutionRun:
    matches = [
        candidate
        for candidate in runs
        if candidate.experiment == experiment and candidate.seed == seed
    ]
    if len(matches) != 1:
        raise FormalRunExecutionError(
            "experiment and seed must identify exactly one run"
        )
    return matches[0]


def inspect_formal_run(
    matrix_path: Path,
    resource_budget_path: Path,
    *,
    experiment: str,
    seed: int,
    authorization_path: Path | None = None,
    protocol_path: Path = Path("configs/experiments/formal_experiment_protocol.yaml"),
    git_commit_reader: GitCommitReader = _current_git_commit,
    worktree_checker: WorktreeChecker = _tracked_worktree_clean,
) -> FormalExecutionRun:
    """Return one validated run without executing its command."""

    plan = load_formal_execution_plan(
        matrix_path,
        resource_budget_path,
        require_unauthorized=True,
    )
    run = _select_run(
        plan.runs,
        experiment=experiment,
        seed=seed,
    )
    if authorization_path is None:
        return run
    approval = _load_mapping(authorization_path)
    commit = _authorization_commit(approval, protocol_id=plan.protocol_id)
    if commit != git_commit_reader():
        raise FormalRunExecutionError("current git commit does not match the authorized commit")
    if not worktree_checker():
        raise FormalRunExecutionError("tracked working tree must be clean")
    if approval.get("runs") != [{"experiment": experiment, "seed": seed}]:
        raise FormalRunExecutionError("authorization must cover exactly this experiment and seed")
    if approval.get("sha256") != _bound_hashes(
        matrix_path, resource_budget_path, protocol_path, run.config,
    ):
        raise FormalRunExecutionError("authorization document hashes do not match")
    protocol = _load_mapping(protocol_path)
    if (
        protocol.get("protocol_id") != plan.protocol_id
        or protocol.get("evaluation", {}).get("selection_split") != "val"
    ):
        raise FormalRunExecutionError("protocol must match and select on val")
    blockers = tuple(b for b in run.blockers if b != "full_training_not_authorized")
    return replace(run, blockers=blockers, ready=not blockers)


def execute_formal_run(
    matrix_path: Path,
    resource_budget_path: Path,
    *,
    experiment: str,
    seed: int,
    confirmation: str,
    authorization_path: Path | None = None,
    protocol_path: Path = Path("configs/experiments/formal_experiment_protocol.yaml"),
    command_runner: CommandRunner = _run_command,
    git_commit_reader: GitCommitReader = _current_git_commit,
    worktree_checker: WorktreeChecker = _tracked_worktree_clean,
) -> FormalRunExecutionResult:
    """Validate every gate and execute one training command only."""

    expected_confirmation = f"{experiment}:{seed}"
    if confirmation != expected_confirmation:
        raise FormalRunExecutionError(
            f"confirmation must exactly equal {expected_confirmation}"
        )
    if not worktree_checker():
        raise FormalRunExecutionError("tracked working tree must be clean")
    if authorization_path is None:
        raise FormalRunExecutionError("formal run is not authorized: --authorization required")
    run = inspect_formal_run(
        matrix_path, resource_budget_path, experiment=experiment, seed=seed,
        authorization_path=authorization_path, protocol_path=protocol_path,
        git_commit_reader=git_commit_reader, worktree_checker=worktree_checker,
    )
    current_commit = git_commit_reader()
    if not run.ready:
        raise FormalRunExecutionError(
            "formal run is blocked: " + ", ".join(run.blockers)
        )

    return_code = command_runner(run.training_argv)
    if return_code != 0:
        raise FormalRunExecutionError(
            f"formal training command failed with exit code {return_code}"
        )
    return FormalRunExecutionResult(
        experiment=run.experiment,
        seed=run.seed,
        config=run.config,
        output_root=run.output_root,
        git_commit=current_commit,
        command=run.training_argv,
        return_code=return_code,
    )
