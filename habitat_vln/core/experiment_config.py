"""Load small, explicit YAML experiment presets for command-line entrypoints."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml

EXPERIMENT_CONFIG_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated metadata and argparse defaults loaded from one YAML file."""

    name: str
    description: str
    arguments: Dict[str, Any]
    path: Path


def load_experiment_config(
    path, valid_arguments: Iterable[str]
) -> ExperimentConfig:
    """Load one YAML preset and reject unsupported command-line arguments."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Experiment config does not exist: {config_path}"
        )

    with config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Experiment config must contain a YAML mapping")

    schema_version = payload.get("schema_version")
    if schema_version != EXPERIMENT_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported experiment config schema_version: "
            f"{schema_version!r}; expected {EXPERIMENT_CONFIG_SCHEMA_VERSION}"
        )

    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            "Experiment config requires a non-empty string 'name'"
        )
    description = payload.get("description", "")
    if not isinstance(description, str):
        raise ValueError("Experiment config 'description' must be a string")

    arguments = payload.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError("Experiment config 'arguments' must be a mapping")

    valid_argument_set = set(valid_arguments)
    unknown_arguments = sorted(set(arguments) - valid_argument_set)
    if unknown_arguments:
        raise ValueError(
            "Unknown experiment arguments: " + ", ".join(unknown_arguments)
        )

    return ExperimentConfig(
        name=name.strip(),
        description=description.strip(),
        arguments=dict(arguments),
        path=config_path,
    )
