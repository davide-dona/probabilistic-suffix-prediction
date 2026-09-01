from __future__ import annotations

from pydantic import Field, model_validator

from .base import NAME_PATTERN, StrictModel


class EmbeddingConfig(StrictModel):
    """Event embeddings, shared by the trace encoder and the decoder."""

    activity_dim: int = Field(..., gt=0)
    resource_dim: int = Field(..., gt=0)
    feature_dim: int = Field(
        ..., gt=0, description='Width of every categorical event feature table, shared across them'
    )


class TraceEncoderConfig(StrictModel):
    """Transformer encoder over a sequence of events, full self-attention.

    One stack reads both sequences: the prefix, whose summary conditions the prior and whose
    events the decoder cross-attends over, and, on the training path only, the ground-truth
    suffix, whose summary feeds the posterior.
    """

    num_layers: int = Field(..., gt=0)
    num_heads: int = Field(..., gt=0)
    feedforward_dim: int = Field(..., gt=0)
    dropout: float = Field(..., ge=0.0, lt=1.0)


class PriorConfig(StrictModel):
    """MLP mapping the prefix summary to p(z | prefix)."""

    hidden_dims: list[int] = Field(..., description='Empty for a linear prior')
    dropout: float = Field(..., ge=0.0, lt=1.0)


class LatentConfig(StrictModel):
    latent_dim: int = Field(..., gt=0)


class DecoderConfig(StrictModel):
    """Transformer decoder writing the suffix: causal self-attention plus cross-attention over
    the encoded prefix.
    """

    num_layers: int = Field(..., gt=0)
    num_heads: int = Field(..., gt=0)
    feedforward_dim: int = Field(..., gt=0)
    dropout: float = Field(..., ge=0.0, lt=1.0)
    activity_dropout: float = Field(
        ...,
        ge=0.0,
        lt=1.0,
        description='Fraction of teacher-forced activities blanked to PAD during training, '
        'forcing information into z instead',
    )
    head_hidden_dim: int = Field(
        ..., gt=0, description='Width of the layer shared by the two output heads'
    )


class ModelConfig(StrictModel):
    """Every hyperparameter of `TransformerCVAE`.

    Data-derived dimensions (vocabulary sizes, special-token indices, sequence length) are
    absent: they come from `DatasetCodec` at build time.
    """

    name: str = Field(..., pattern=NAME_PATTERN)
    d_model: int = Field(
        ..., gt=0, description='Shared width for the embeddings, the encoder and the decoder'
    )

    embeddings: EmbeddingConfig
    encoder: TraceEncoderConfig
    prior: PriorConfig
    latent: LatentConfig
    decoder: DecoderConfig

    @model_validator(mode='after')
    def _heads_divide_width(self) -> ModelConfig:
        # nn.MultiheadAttention asserts this when the layer is built, halfway through a run's
        # setup. Checking it here turns a config mistake back into a config error.
        for name in ('encoder', 'decoder'):
            num_heads = getattr(self, name).num_heads
            if self.d_model % num_heads != 0:
                raise ValueError(
                    f'model.{name}.num_heads ({num_heads}) must divide '
                    f'model.d_model ({self.d_model})'
                )
        return self
