from __future__ import annotations

from pydantic import Field, model_validator

from .base import NAME_PATTERN, StrictModel


class EmbeddingConfig(StrictModel):
    """Event embeddings, shared by both trace encoders and the decoder."""

    activity_dim: int = Field(..., gt=0)
    resource_dim: int = Field(..., gt=0)
    feature_dim: int = Field(
        ...,
        gt=0,
        description="Width of each categorical event feature's embedding table, shared across "
        'all of them: the more features, the more a bigger value widens the projection input',
    )


class TraceEncoderConfig(StrictModel):
    """Transformer encoder over a sequence of events, full self-attention.

    Reads the prefix, whose summary conditions the prior and whose events the decoder
    cross-attends over. Runs at `ModelConfig.d_model`.
    """

    num_layers: int = Field(..., gt=0)
    num_heads: int = Field(
        ..., gt=0, description='Attention heads per layer; must divide `d_model`'
    )
    feedforward_dim: int = Field(
        ..., gt=0, description='Width of the feed-forward block inside a layer'
    )
    dropout: float = Field(..., ge=0.0, lt=1.0)


class PriorConfig(StrictModel):
    """MLP mapping the prefix summary to p(k | prefix), the distribution over the latent's
    modes.
    """

    hidden_dims: list[int] = Field(..., description='Hidden layer widths; empty for a linear prior')
    dropout: float = Field(..., ge=0.0, lt=1.0)


class LatentConfig(StrictModel):
    """The modes the latent chooses between, one of which the decoder writes each suffix under."""

    num_modes: int = Field(
        ...,
        gt=1,
        description='How many ways the model may continue one prefix. Every training step writes '
        "the suffix under all of them, so it is the factor the decoder's work is multiplied by",
    )


class DecoderConfig(StrictModel):
    """Transformer decoder writing the suffix: causal self-attention plus cross-attention over
    the encoded prefix. Runs at `ModelConfig.d_model`.
    """

    num_layers: int = Field(..., gt=0)
    num_heads: int = Field(
        ..., gt=0, description='Attention heads per layer; must divide `d_model`'
    )
    feedforward_dim: int = Field(
        ..., gt=0, description='Width of the feed-forward block inside a layer'
    )
    dropout: float = Field(..., ge=0.0, lt=1.0)
    activity_dropout: float = Field(
        ...,
        ge=0.0,
        lt=1.0,
        description='Fraction of teacher-forced activities blanked to PAD during training, '
        'forcing information into z instead. 0.0 disables it',
    )
    head_hidden_dim: int = Field(
        ..., gt=0, description='Width of the layer shared by the two output heads'
    )


class ModelConfig(StrictModel):
    """Every hyperparameter of `TransformerCVAE`.

    Data-derived dimensions (vocabulary sizes, special-token indices, sequence length) are
    absent: they come from `DatasetCodec` at build time.
    """

    name: str = Field(
        ...,
        pattern=NAME_PATTERN,
        description='What this architecture is called, e.g. `cvae`. Names the runs it produces, '
        'and is what a figure tells two models apart by',
    )

    d_model: int = Field(
        ..., gt=0, description='Shared width for the embeddings, the prefix encoder and the decoder'
    )

    embeddings: EmbeddingConfig
    prefix_encoder: TraceEncoderConfig
    prior: PriorConfig
    latent: LatentConfig
    decoder: DecoderConfig

    @model_validator(mode='after')
    def _heads_divide_width(self) -> ModelConfig:
        # nn.MultiheadAttention asserts this when the layer is built, halfway through a run's
        # setup. Checking it here turns a config mistake back into a config error.
        for name in ('prefix_encoder', 'decoder'):
            num_heads = getattr(self, name).num_heads
            if self.d_model % num_heads != 0:
                raise ValueError(
                    f'model.{name}.num_heads ({num_heads}) must divide '
                    f'model.d_model ({self.d_model})'
                )
        return self
