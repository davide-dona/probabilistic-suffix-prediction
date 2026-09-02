from pydantic import Field

from .base import StrictModel


class DataLoaderConfig(StrictModel):
    """torch.utils.data.DataLoader parameters."""

    batch_size: int = Field(..., gt=0)
    num_workers: int = Field(..., ge=0)


class LossConfig(StrictModel):
    """KL weighting: the annealing ramp and the free-bits floor.

    The ramp is a number of optimizer steps rather than a fraction of `training.max_steps`, so it
    means the same thing on every dataset and a run that stops early has still spent the same
    time reaching full weight.
    """

    kl_annealing_ramp_steps: int = Field(..., gt=0)
    kl_annealing_start_weight: float = Field(..., ge=0.0)
    kl_annealing_full_weight: float = Field(..., ge=0.0)
    free_bits: float = Field(
        ..., ge=0.0, description='Nats per latent dimension the KL is not penalized below'
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
        None, gt=0.0, description='Null or absent leaves gradients unclipped'
    )
    device: str = Field(..., pattern=r'^(cpu|mps|cuda(:\d+)?)$')
    val_every_n_steps: int = Field(
        ..., gt=0, description='Also the unit `early_stopping.patience` counts in'
    )
    validation_pairs: int = Field(
        ..., gt=0, description='Teacher-forced pairs read per validation, capped by split size'
    )
    generation_pairs: int = Field(
        ...,
        gt=0,
        description='Free-running prefixes per validation; the sample the selection score is '
        'computed on',
    )


class EarlyStoppingConfig(StrictModel):
    """Stops training once the free-running generation score plateaus, the same score used for
    best-model selection.
    """

    patience: int = Field(
        ..., gt=0, description='Non-improving validations tolerated before stopping'
    )
    min_delta_perc: float = Field(
        ..., ge=0.0, description='Minimum relative improvement to reset the patience counter'
    )
