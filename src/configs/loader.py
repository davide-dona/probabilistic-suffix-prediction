from pathlib import Path

import yaml

from src.paths import BASE_CONFIG

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


def _read(path: Path) -> dict:
    """Read one layer of a config."""
    with path.open('r') as f:
        return yaml.safe_load(f)


def load_config(config: Path, hardware: Path) -> ExperimentConfig:
    """Load and validate an experiment config.
    Merges with the base config and the selected hardware profile.

    Args:
        config: Path to the dataset config YAML, e.g. config/datasets/bpic17.yaml.
        hardware: Path to the hardware profile YAML, e.g. config/hardware/mps.yaml.
    Returns:
        The validated config.
    """
    # Load the base config, the hardware profile, and the dataset.
    # Overrides are applied in that same order.
    merged = _deep_merge(_deep_merge(_read(BASE_CONFIG), _read(hardware)), _read(config))
    return ExperimentConfig.model_validate(merged)


def load_dataset_config(config: Path) -> DatasetConfig:
    """Load and validate the hardware-independent parts of an experiment config: `data` and
    `declare`, merged over `paths.BASE_CONFIG` alone. For pipelines that never read a
    hardware-dependent value, so they never need a hardware profile.

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

    The run's own config travels inside the checkpoint, so generating needs no dataset config: the
    model, the data and the sampling it was trained under are already settled. Only the profile
    changes, since a run trained on one machine is routinely generated from on another.

    The profile carries training-only values too - `training.max_steps`, `optimizer.lr`,
    `loss.kl_annealing_period_steps` - and merging brings them along. Nothing generation reads
    touches them and the merged config is never written back out, so they go no further than here.

    Args:
        experiment_config: The run's config as the checkpoint stores it, from
            `checkpoint['experiment_config']`.
        hardware: Path to the hardware profile to generate under, whose device, batch size,
            workers and row bound replace the ones the run was trained with.
        num_samples: How many suffixes to draw per prefix, or `None` to keep the run's own.
    Returns:
        The validated config.
    Raises:
        pydantic.ValidationError: If the merged config is not a valid experiment, `num_samples`
            below `InferenceConfig`'s floor included.
    """
    merged = _deep_merge(experiment_config, _read(hardware))
    if num_samples is not None:
        # Merged in rather than set on the config, which is frozen once validated.
        merged = _deep_merge(merged, {'inference': {'num_samples': num_samples}})
    return ExperimentConfig.model_validate(merged)
