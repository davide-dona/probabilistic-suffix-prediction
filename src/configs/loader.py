from pathlib import Path

import yaml

from src.paths import BASE_CONFIG, hardware_config_path

from .schema import DatasetConfig, ExperimentConfig


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge two dicts, with override taking precedence.
    Recursion allows nested dicts to be merged rather than replaced, so a config can override
    just one field of a nested section.
    Args:
        base: The base config dict.
        override: The override config dict.
    Returns:
        The merged config dict.
    """
    # Start with a copy of the base dict
    merged = dict(base)

    # For each key/value pair in the override dict
    for key, value in override.items():
        # If both are dicts, merge them recursively
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            # Otherwise, override the base value with the override value
            merged[key] = value
    return merged


def load_config(path: str | Path, hardware: str) -> ExperimentConfig:
    """Load and validate an experiment config.
    Merges with the base config and the selected hardware profile.

    Args:
        path: The dataset config YAML, under config/datasets/.
        hardware: Name of the hardware profile YAML, under config/hardware/ (e.g. 'mps').
    Returns:
        The validated config.
    """
    path = Path(path)
    # Load the base config, the hardware profile, and the dataset.
    # Overrides are applied in that same order.
    with BASE_CONFIG.open('r') as f:
        base = yaml.safe_load(f)
    with hardware_config_path(hardware).open('r') as f:
        hardware_profile = yaml.safe_load(f)
    with path.open('r') as f:
        override = yaml.safe_load(f)

    merged = _deep_merge(_deep_merge(base, hardware_profile), override)
    return ExperimentConfig.model_validate(merged)


def load_dataset_config(path: str | Path) -> DatasetConfig:
    """Load and validate the hardware-independent parts of an experiment config: `data` and
    `declare`, merged over `paths.BASE_CONFIG` alone. For pipelines that never read a
    hardware-dependent value, so they never need a hardware profile.

    Args:
        path: The dataset config YAML, under config/datasets/.
    Returns:
        The validated `data`/`declare` sections.
    """
    path = Path(path)
    with BASE_CONFIG.open('r') as f:
        base = yaml.safe_load(f)
    with path.open('r') as f:
        override = yaml.safe_load(f)
    merged = _deep_merge(base, override)
    return DatasetConfig.model_validate({'data': merged['data'], 'declare': merged['declare']})
