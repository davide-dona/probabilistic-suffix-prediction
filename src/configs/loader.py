from pathlib import Path

import yaml

from src.paths import BASE_CONFIG

from .schema import DatasetConfig, ExperimentConfig


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge two dicts, with override taking precedence.
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


def _read(path: Path) -> dict:
    """Read one layer of a config."""
    with path.open('r') as f:
        return yaml.safe_load(f)


def load_config(model: Path, config: Path, hardware: Path) -> ExperimentConfig:
    """Load and validate an experiment config, merging the base config, the model, the hardware
    profile and the dataset config in that order.

    The model is a layer of its own because an architecture is chosen independently of the log it
    is run on: one file per architecture against one per dataset, rather than a copy of each
    model's widths in every dataset config.

    Args:
        model: Path to the model config YAML, e.g. config/models/cvae.yaml. Its `model.kind` is
            what says which architecture the run builds.
        config: Path to the dataset config YAML, e.g. config/datasets/bpic17.yaml.
        hardware: Path to the hardware profile YAML, e.g. config/hardware/cuda-a6000.yaml.
    Returns:
        The validated config.
    """
    # Load the base config, the model, the hardware profile, and the dataset.
    # Overrides are applied in that same order.
    merged = _deep_merge(_read(BASE_CONFIG), _read(model))
    merged = _deep_merge(_deep_merge(merged, _read(hardware)), _read(config))
    return ExperimentConfig.model_validate(merged)


def load_dataset_config(config: Path) -> DatasetConfig:
    """Load and validate the hardware-independent parts of an experiment config.
    Args:
        config: Path to the dataset config YAML, e.g. config/datasets/bpic17.yaml.
    Returns:
        The validated `data`/`declare` sections.
    """
    merged = _deep_merge(_read(BASE_CONFIG), _read(config))
    return DatasetConfig.model_validate({'data': merged['data'], 'declare': merged['declare']})


def load_generation_config(
    experiment_config: dict, *, hardware: Path, num_samples: int | None
) -> ExperimentConfig:
    """Load and validate the config a checkpoint is generated from, under the hardware in hand.
    Args:
        experiment_config: The run's config as the checkpoint stores it, from
            `checkpoint['experiment_config']`.
        hardware: Path to the hardware profile to generate under, whose device, batch size,
            workers and row bound replace the ones the run was trained with.
        num_samples: How many suffixes to draw per prefix, replacing the run's own
            `inference.evaluation_samples`, or `None` to keep it.
    Returns:
        The validated config.
    Raises:
        pydantic.ValidationError: If the merged config is not a valid experiment, `num_samples`
            below `InferenceConfig`'s floor included.
    """
    merged = _deep_merge(experiment_config, _read(hardware))
    if num_samples is not None:
        # Merged in rather than set on the config, which is frozen once validated.
        merged = _deep_merge(merged, {'inference': {'evaluation_samples': num_samples}})
    return ExperimentConfig.model_validate(merged)
