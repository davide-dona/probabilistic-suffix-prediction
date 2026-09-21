import re
from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.validation.data import validate_data, validate_declare
from src.validation.model import validate_model, validate_sampling
from src.validation.primitives import validate_number, validate_string_list


def validate_preprocess(config: DictConfig) -> None:
    """Validate all static parameters accepted by the preprocessing stage."""
    validate_data(config.data)
    validate_declare(config.declare)


def validate_training(config: DictConfig) -> None:
    """Validate the complete effective experiment configuration used by training.

    Raises:
        ValueError: If a data, model, optimizer, runtime, inference, or tracking setting is
            invalid or inconsistent with another setting.
    """
    validate_preprocess(config)
    validate_model(config.model)

    training = config.training
    dataloader = config.dataloader
    optimizer = config.optimizer
    early_stopping = config.early_stopping
    inference = config.inference

    validate_number(config.seed, 'seed', inclusive=True, integer=True)

    for key in ('max_steps', 'val_every_n_steps', 'validation_pairs', 'generation_pairs'):
        validate_number(training[key], f'training.{key}', integer=True)
    if training.grad_clip_norm is not None:
        validate_number(training.grad_clip_norm, 'training.grad_clip_norm')
    device_is_valid = (
        isinstance(training.device, str)
        and re.fullmatch(r'(cpu|mps|cuda(:\d+)?)', training.device) is not None
    )
    if not device_is_valid:
        raise ValueError('training.device must be cpu, mps, cuda, or cuda:<index>')

    validate_number(dataloader.batch_size, 'dataloader.batch_size', integer=True)
    validate_number(dataloader.num_workers, 'dataloader.num_workers', inclusive=True, integer=True)

    validate_number(optimizer.lr, 'optimizer.lr')
    validate_number(optimizer.weight_decay, 'optimizer.weight_decay', inclusive=True)
    validate_number(optimizer.warmup_steps, 'optimizer.warmup_steps', inclusive=True, integer=True)

    validate_number(early_stopping.patience, 'early_stopping.patience', integer=True)
    validate_number(early_stopping.min_delta_perc, 'early_stopping.min_delta_perc', inclusive=True)

    for key in ('validation_samples', 'evaluation_samples'):
        validate_number(
            inference[key],
            f'inference.{key}',
            minimum=10,
            inclusive=True,
            integer=True,
        )
    validate_number(
        inference.generation_rows_upper_bound,
        'generation_rows_upper_bound',
        integer=True,
    )
    if inference.generation_rows_upper_bound < max(
        inference.validation_samples, inference.evaluation_samples
    ):
        raise ValueError('generation_rows_upper_bound must fit at least one prefix of samples')

    if config.wandb.mode not in ('online', 'offline', 'disabled'):
        raise ValueError('wandb.mode must be online, offline, or disabled')


def validate_generation(
    config: DictConfig, *, tuning: Path | None, sampling: DictConfig | None
) -> None:
    """Validate effective generation configuration and optional sampler overrides.

    A tuning report and direct sampler are mutually exclusive. Direct sampler overrides are
    available only to the head-sampling Transformer.
    """
    if tuning is not None and sampling is not None:
        raise ValueError('Choose either tuning or sampling overrides')

    if sampling is not None:
        validate_sampling(sampling)

    has_sampling_override = tuning is not None or sampling is not None
    if has_sampling_override and config.model.kind != 'head_sampling_transformer':
        raise ValueError('Only head_sampling_transformer supports sampling overrides')

    validate_training(config)


def validate_tuning(
    config: DictConfig, *, temperatures: Sequence[object], top_ps: Sequence[object]
) -> None:
    """Validate effective tuning configuration and the Cartesian sampler grid."""
    validate_training(config)

    if config.model.kind != 'head_sampling_transformer':
        raise ValueError(
            f'{config.model.kind} has no output-head sampler to tune. Transformer CVAE draws '
            'its variability from z instead.'
        )

    if not temperatures or not top_ps:
        raise ValueError('Sampler grid cannot be empty')

    for temperature in temperatures:
        for top_p in top_ps:
            validate_sampling(OmegaConf.create({'temperature': temperature, 'top_p': top_p}))


def validate_evaluation(*, workers: object) -> None:
    """Validate optional process-pool sizing for evaluation."""
    if workers is not None:
        validate_number(workers, 'workers', integer=True)


def validate_visualization(*, evaluations: object, evaluations_dir: object) -> None:
    """Validate the mutually exclusive report-source parameters for visualization."""
    validate_string_list(evaluations, 'evaluations')
    validate_string_list(evaluations_dir, 'evaluations_dir')
    if bool(evaluations) == bool(evaluations_dir):
        raise ValueError('Provide evaluations or evaluations_dir, exclusively')
