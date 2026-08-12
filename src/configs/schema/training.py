from typing import Literal

from pydantic import Field

from .base import StrictModel


class DataLoaderConfig(StrictModel):
    """torch.utils.data.DataLoader parameters; hardware-dependent, owned by the hardware
    profile rather than the dataset.
    """

    batch_size: int = Field(..., gt=0)
    num_workers: int = Field(..., ge=0)


class LossConfig(StrictModel):
    """KL weighting: the annealing schedule and the free-bits floor.

    Both are measured in optimizer steps, not epochs, so they mean the same thing on every
    dataset. The cycle is a length rather than a count fitted into `training.max_steps`, so a
    run that stops early has still seen whole cycles.
    """

    kl_annealing_period_steps: int = Field(..., gt=0, description='Optimizer steps in one cycle')
    kl_annealing_ratio: float = Field(
        ..., gt=0.0, lt=1.0, description='Fraction of each cycle spent ramping up'
    )
    kl_annealing_start_weight: float = Field(
        ..., ge=0.0, description='Weight each cycle ramps up from'
    )
    kl_annealing_full_weight: float = Field(
        ..., ge=0.0, description='Weight each cycle ramps up to, and holds at'
    )

    free_bits: float = Field(
        ...,
        ge=0.0,
        description='Nats per latent dimension the KL is not penalized below. 0.0 leaves it '
        'unfloored',
    )


class OptimizerConfig(StrictModel):
    lr: float = Field(..., gt=0.0)
    weight_decay: float = Field(..., ge=0.0)


class TrainingConfig(StrictModel):
    """How long a run goes on for and how often it validates, both counted in optimizer steps
    rather than epochs, since an epoch is a different amount of learning on every log.
    """

    max_steps: int = Field(
        ...,
        gt=0,
        description='Ceiling on optimizer steps; early stopping normally ends a run first',
    )
    grad_clip_norm: float | None = Field(
        None, gt=0.0, description='Max gradient norm; null or absent leaves gradients unclipped'
    )
    device: Literal['cpu', 'cuda', 'mps']
    val_every_n_steps: int = Field(
        ...,
        gt=0,
        description='Steps between validations; also the unit `early_stopping.patience` counts in',
    )
    validation_pairs: int = Field(
        ...,
        gt=0,
        description='Teacher-forced validation pairs read per validation, capped by split size',
    )
    generation_pairs: int = Field(
        ...,
        gt=0,
        description='Free-running generation prefixes per validation; the sample the selection '
        'score is computed on',
    )


class EarlyStoppingConfig(StrictModel):
    """Stops training once the free-running generation score plateaus, the same score used for
    best-model selection and reported by `pipelines/evaluate.py`.
    """

    patience: int = Field(
        ..., gt=0, description='Non-improving validations tolerated before stopping'
    )
    min_delta_perc: float = Field(
        ..., ge=0.0, description='Minimum relative improvement to reset the patience counter'
    )
