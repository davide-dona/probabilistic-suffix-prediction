from pydantic import Field

from .base import StrictModel


class InferenceConfig(StrictModel):
    """Generating suffixes for a whole split, which is what evaluation reads. Runs on
    `training.device`.
    """

    num_samples: int = Field(
        ...,
        ge=10,
        description='Suffixes generated per prefix from p(z | prefix). Minimum 10: '
        '`hit_rate_at_10` reads the tenth draw',
    )
    generation_rows_upper_bound: int = Field(
        ...,
        gt=0,
        description='Upper bound on rows put through the decoder per call, where one prefix is '
        'num_samples rows. Bounds decoder memory independently of num_samples and of the '
        'DataLoader batch size, which `generation_batch_size()` weighs against it',
    )
