"""Audit a formal experiment configuration matrix."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class FormalExperimentAuditError(
    ValueError
):
    """Raised when a formal experiment matrix is invalid."""


@dataclass(frozen=True)
class FormalExperimentAudit:
    """Successful formal experiment matrix audit result."""

    protocol_id: str
    run_count: int
    experiments: tuple[str, ...]
    seeds: tuple[int, ...]
    config_sha256: dict[str, str]


def _load_mapping(
    path: Path,
    *,
    label: str,
) -> tuple[dict[str, Any], bytes]:
    source = Path(path)

    if not source.is_file():
        raise FormalExperimentAuditError(
            f"{label} must be a regular file: {source}"
        )

    try:
        raw = source.read_bytes()
        text = raw.decode("utf-8")
        payload = yaml.safe_load(text)
    except (
        OSError,
        UnicodeError,
        yaml.YAMLError,
    ) as exc:
        raise FormalExperimentAuditError(
            f"failed to load {label}: {source}"
        ) from exc

    if not isinstance(payload, dict):
        raise FormalExperimentAuditError(
            f"{label} root must be a mapping"
        )

    return payload, raw


def _require_non_empty_string(
    payload: Mapping[str, Any],
    field: str,
    *,
    label: str,
) -> str:
    value = payload.get(field)

    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise FormalExperimentAuditError(
            f"{label} {field} must be a non-empty string"
        )

    return value.strip()


def _require_positive_integer(
    payload: Mapping[str, Any],
    field: str,
    *,
    label: str,
) -> int:
    value = payload.get(field)

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise FormalExperimentAuditError(
            f"{label} {field} must be a positive integer"
        )

    return value


def _normalized_path(
    value: str,
) -> str:
    return Path(value).as_posix().rstrip("/")


def audit_formal_experiment_matrix(
    protocol_path: Path,
    matrix_path: Path,
) -> FormalExperimentAudit:
    """Validate formal run coverage and common config contracts."""

    protocol, _ = _load_mapping(
        Path(protocol_path),
        label="protocol",
    )
    matrix, _ = _load_mapping(
        Path(matrix_path),
        label="matrix",
    )

    if protocol.get("format_version") != 1:
        raise FormalExperimentAuditError(
            "protocol format_version must be 1"
        )

    if protocol.get("status") != "frozen":
        raise FormalExperimentAuditError(
            "protocol status must be frozen"
        )

    protocol_id = _require_non_empty_string(
        protocol,
        "protocol_id",
        label="protocol",
    )

    execution_gate = protocol.get(
        "execution_gate"
    )
    if (
        not isinstance(
            execution_gate,
            Mapping,
        )
        or execution_gate.get(
            "full_training_authorized"
        )
        is not False
    ):
        raise FormalExperimentAuditError(
            "execution gate must remain closed"
        )

    dataset = protocol.get("dataset")
    if not isinstance(dataset, Mapping):
        raise FormalExperimentAuditError(
            "protocol dataset must be a mapping"
        )

    expected_data_root = (
        _require_non_empty_string(
            dataset,
            "root",
            label="protocol dataset",
        )
    )

    training = protocol.get("training")
    if not isinstance(training, Mapping):
        raise FormalExperimentAuditError(
            "protocol training must be a mapping"
        )

    expected_epochs = (
        _require_positive_integer(
            training,
            "epochs",
            label="protocol training",
        )
    )

    raw_seeds = training.get("seeds")
    if (
        not isinstance(raw_seeds, list)
        or not raw_seeds
    ):
        raise FormalExperimentAuditError(
            "protocol training seeds must be a non-empty list"
        )

    seeds: list[int] = []
    for index, seed in enumerate(raw_seeds):
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
        ):
            raise FormalExperimentAuditError(
                "protocol training seed "
                f"{index} must be a non-negative integer"
            )
        seeds.append(seed)

    if len(set(seeds)) != len(seeds):
        raise FormalExperimentAuditError(
            "protocol training seeds must be unique"
        )

    raw_experiments = protocol.get(
        "experiments"
    )
    if (
        not isinstance(
            raw_experiments,
            Mapping,
        )
        or not raw_experiments
    ):
        raise FormalExperimentAuditError(
            "protocol experiments must be a non-empty mapping"
        )

    experiments: list[str] = []
    for experiment, specification in raw_experiments.items():
        if (
            not isinstance(experiment, str)
            or not experiment.strip()
        ):
            raise FormalExperimentAuditError(
                "protocol experiment names must be non-empty strings"
            )
        if not isinstance(specification, Mapping):
            raise FormalExperimentAuditError(
                "protocol experiment specifications must be mappings"
            )
        experiments.append(
            experiment.strip()
        )

    if matrix.get("format_version") != 1:
        raise FormalExperimentAuditError(
            "matrix format_version must be 1"
        )

    if matrix.get("protocol_id") != protocol_id:
        raise FormalExperimentAuditError(
            "matrix protocol_id does not match protocol"
        )

    runs = matrix.get("runs")
    if not isinstance(runs, list):
        raise FormalExperimentAuditError(
            "matrix runs must be a list"
        )

    expected_pairs = {
        (experiment, seed)
        for experiment in experiments
        for seed in seeds
    }
    observed_pairs: set[
        tuple[str, int]
    ] = set()

    experiment_ids: set[str] = set()
    output_roots: set[str] = set()
    config_hashes: dict[str, str] = {}
    configs_by_pair: dict[
        tuple[str, int],
        dict[str, Any],
    ] = {}

    for index, run in enumerate(runs):
        label = f"matrix run {index}"

        if not isinstance(run, Mapping):
            raise FormalExperimentAuditError(
                f"{label} must be a mapping"
            )

        experiment = (
            _require_non_empty_string(
                run,
                "experiment",
                label=label,
            )
        )
        if experiment not in experiments:
            raise FormalExperimentAuditError(
                f"{label} experiment is not declared in protocol"
            )

        seed = run.get("seed")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed not in seeds
        ):
            raise FormalExperimentAuditError(
                f"{label} seed is not declared in protocol"
            )

        pair = (experiment, seed)
        if pair in observed_pairs:
            raise FormalExperimentAuditError(
                "duplicate matrix experiment and seed"
            )
        observed_pairs.add(pair)

        config_value = (
            _require_non_empty_string(
                run,
                "config",
                label=label,
            )
        )
        config_path = Path(config_value)
        config, config_bytes = (
            _load_mapping(
                config_path,
                label=f"{label} config",
            )
        )
        configs_by_pair[pair] = config

        config_seed = config.get("seed")
        if config_seed != seed:
            raise FormalExperimentAuditError(
                f"{label} config seed does not match matrix seed"
            )

        config_epochs = config.get(
            "epochs"
        )
        if config_epochs != expected_epochs:
            raise FormalExperimentAuditError(
                f"{label} config epochs do not match protocol"
            )

        is_unimodal = experiment in {
            "rgb_only",
            "ir_only",
        }

        if is_unimodal:
            if "resume" in config:
                raise FormalExperimentAuditError(f"{label} unimodal config must not contain resume")

            if config.get("pause_after_epoch") is not None:
                raise FormalExperimentAuditError(f"{label} config pause_after_epoch must be null")

            yaml_paths = dataset.get("yaml_paths")
            if not isinstance(
                yaml_paths,
                Mapping,
            ):
                raise FormalExperimentAuditError("protocol dataset yaml_paths must be a mapping")

            modality = "rgb" if experiment == "rgb_only" else "ir"
            expected_data = _require_non_empty_string(
                yaml_paths,
                modality,
                label=("protocol dataset yaml_paths"),
            )
            declared_data = _require_non_empty_string(
                config,
                "data",
                label=f"{label} config",
            )

            if _normalized_path(declared_data) != _normalized_path(expected_data):
                raise FormalExperimentAuditError(
                    f"{label} config data does not match the protocol {modality} YAML"
                )

            prediction_source = _require_non_empty_string(
                config,
                "prediction_source",
                label=f"{label} config",
            )
            expected_prediction_source = f"{expected_data_root}/{modality}/images/val"

            if _normalized_path(prediction_source) != _normalized_path(expected_prediction_source):
                raise FormalExperimentAuditError(
                    f"{label} config prediction_source must be the validation image directory"
                )
        else:
            if config.get("resume") is not False:
                raise FormalExperimentAuditError(f"{label} config resume must be false")

            data_root = _require_non_empty_string(
                config,
                "data_root",
                label=f"{label} config",
            )
            if _normalized_path(data_root) != _normalized_path(expected_data_root):
                raise FormalExperimentAuditError(
                    f"{label} config data_root does not match protocol"
                )

            if "yolo_smoke_5pct" in (_normalized_path(data_root)):
                raise FormalExperimentAuditError(f"{label} config data_root references smoke data")

        experiment_id = (
            _require_non_empty_string(
                config,
                "experiment_id",
                label=f"{label} config",
            )
        )
        if experiment_id in experiment_ids:
            raise FormalExperimentAuditError(
                "duplicate experiment_id"
            )
        experiment_ids.add(experiment_id)
        experiment_slug = experiment.replace(
            "_",
            "-",
        )
        expected_experiment_id = (
            f"formal-{experiment_slug}-seed{seed}"
        )
        if experiment_id != expected_experiment_id:
            raise FormalExperimentAuditError(
                f"{label} config experiment_id must be "
                f"{expected_experiment_id}"
            )

        experiment_group = _require_non_empty_string(
            config,
            "experiment_group",
            label=f"{label} config",
        )
        expected_experiment_group = (
            f"formal-{experiment_slug}"
        )
        if experiment_group != expected_experiment_group:
            raise FormalExperimentAuditError(
                f"{label} config experiment_group must be "
                f"{expected_experiment_group}"
            )

        output_root = (
            _require_non_empty_string(
                config,
                "output_root",
                label=f"{label} config",
            )
        )
        normalized_output_root = (
            _normalized_path(output_root)
        )
        if normalized_output_root in output_roots:
            raise FormalExperimentAuditError(
                "duplicate output_root"
            )
        output_roots.add(
            normalized_output_root
        )
        expected_output_root = (
            f"runs/formal/{experiment}_seed{seed}"
        )
        if normalized_output_root != expected_output_root:
            raise FormalExperimentAuditError(
                f"{label} config output_root must be "
                f"{expected_output_root}"
            )

        config_hashes[
            str(config_path)
        ] = hashlib.sha256(
            config_bytes
        ).hexdigest()

    if observed_pairs != expected_pairs:
        missing = sorted(
            expected_pairs - observed_pairs
        )
        unexpected = sorted(
            observed_pairs - expected_pairs
        )
        raise FormalExperimentAuditError(
            "matrix coverage does not match protocol: "
            f"missing={missing}, unexpected={unexpected}"
        )

    teacher_kinds = {
        "early_fusion_teacher": "official_early_fusion",
        "gated_teacher": "official_gated",
    }

    for experiment, specification in raw_experiments.items():
        teacher_experiment = specification.get("teacher")
        if teacher_experiment is None:
            continue

        if (
            not isinstance(teacher_experiment, str)
            or not teacher_experiment.strip()
            or teacher_experiment not in raw_experiments
        ):
            raise FormalExperimentAuditError(
                f"protocol experiment {experiment} teacher "
                "must name a declared experiment"
            )

        expected_teacher_kind = teacher_kinds.get(
            teacher_experiment
        )
        if expected_teacher_kind is None:
            raise FormalExperimentAuditError(
                f"protocol experiment {experiment} teacher "
                "is not a supported teacher experiment"
            )

        for seed in seeds:
            student_config = configs_by_pair[
                (experiment, seed)
            ]
            teacher_config = configs_by_pair[
                (teacher_experiment, seed)
            ]
            teacher_output_root = _require_non_empty_string(
                teacher_config,
                "output_root",
                label=(
                    f"{teacher_experiment} seed {seed} config"
                ),
            )
            expected_teacher_checkpoint = _normalized_path(
                f"{teacher_output_root}/best_map50_95.pt"
            )
            declared_teacher_checkpoint = (
                _require_non_empty_string(
                    student_config,
                    "teacher_checkpoint",
                    label=f"{experiment} seed {seed} config",
                )
            )

            if _normalized_path(
                declared_teacher_checkpoint
            ) != expected_teacher_checkpoint:
                raise FormalExperimentAuditError(
                    f"{experiment} seed {seed} config "
                    "teacher_checkpoint must reference the same-seed "
                    f"{teacher_experiment} best_map50_95 checkpoint"
                )

            declared_teacher_kind = _require_non_empty_string(
                student_config,
                "teacher_kind",
                label=f"{experiment} seed {seed} config",
            )
            if declared_teacher_kind != expected_teacher_kind:
                raise FormalExperimentAuditError(
                    f"{experiment} seed {seed} config teacher_kind "
                    f"must be {expected_teacher_kind}"
                )

    return FormalExperimentAudit(
        protocol_id=protocol_id,
        run_count=len(runs),
        experiments=tuple(experiments),
        seeds=tuple(seeds),
        config_sha256=config_hashes,
    )
