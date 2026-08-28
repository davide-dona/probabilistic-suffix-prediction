from pydantic import Field

from .base import StrictModel


class InferenceConfig(StrictModel):
    """How many suffixes a prefix is answered with on each of the two passes that generate them,
    and the decoder memory bound the two share. Both run on `training.device`.

    The two are separate because they are paid for on different schedules: the validation pass
    runs inside the training loop and the evaluation pass runs once over the whole test split, so
    the draws a report is built from need not be limited by what a training run can afford every
    `training.val_every_n_steps` steps.
    """

    validation_samples: int = Field(
        ...,
        ge=10,
        description='Suffixes generated per prefix from p(z | prefix) on the in-training '
        'generation pass, over `training.generation_pairs` prefixes. Minimum 10: '
        '`hit_rate_at_10` reads the tenth draw',
    )
    evaluation_samples: int = Field(
        ...,
        ge=10,
        description='Suffixes generated per prefix from p(z | prefix) by `pipelines.generate`, '
        'over the whole test split, which is what a report is built from. Minimum 10: '
        '`hit_rate_at_10` reads the tenth draw',
    )
    generation_rows_upper_bound: int = Field(
        ...,
        gt=0,
        description='Upper bound on rows put through the decoder per call, where one prefix is '
        'as many rows as its pass draws samples. Bounds decoder memory independently of that '
        'sample count and of the DataLoader batch size, which `generation_batch_size()` weighs '
        'against it',
    )
