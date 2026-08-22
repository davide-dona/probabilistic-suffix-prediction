from pydantic import Field

from .base import StrictModel


class DataLoaderConfig(StrictModel):
    """torch.utils.data.DataLoader parameters; hardware-dependent, owned by the hardware
    profile rather than the dataset.
    """

    batch_size: int = Field(..., gt=0)
    num_workers: int = Field(..., ge=0)


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
    device: str = Field(
        ...,
        pattern=r'^(cpu|mps|cuda(:\d+)?)$',
        description='Torch device. `cuda:<n>` pins one GPU on a multi-GPU machine; bare `cuda` '
        'takes whatever the current device is',
    )
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
