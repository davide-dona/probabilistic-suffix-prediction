import torch
from torch import nn

from src.configs.schema import LatentConfig, PoolingConfig, PriorConfig
from src.distributions.gaussian import Gaussian
from src.model.components.pooling import AttentionPooling
from src.model.components.trace_encoder import EncodedTrace


class PriorNetwork(nn.Module):
    """`p(z | prefix)`: the distribution the latent is drawn from at inference time.

    It stands where the fixed `N(0, I)` stands in an unconditional VAE, its mean and variance
    produced from the prefix instead. Because the KL term measures the posterior against it,
    whatever the prefix already determines costs nothing to encode.

    It reads the prefix through a pooling of its own, so what reaches the MLP is pooled for
    `p(z | prefix)` rather than for the encoder, alongside the encoder's own summary of it.
    """

    def __init__(
        self,
        config: PriorConfig,
        pooling_config: PoolingConfig,
        latent_config: LatentConfig,
        *,
        d_model: int,
    ):
        super().__init__()
        self.pooling = AttentionPooling(config=pooling_config, d_model=d_model)

        layers: list[nn.Module] = []
        # The input is the prefix summary beside its pooled rows; each hidden layer then narrows
        # or widens from there.
        width = d_model + self.pooling.pooled_dim
        for hidden_dim in config.hidden_dims:
            layers += [
                nn.Linear(in_features=width, out_features=hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=config.dropout),
            ]
            width = hidden_dim
        # The output layer emits mean and log-variance side by side, hence twice `latent_dim`.
        layers.append(nn.Linear(in_features=width, out_features=2 * latent_config.latent_dim))

        # With `hidden_dims` empty this collapses to a single linear layer.
        self.net = nn.Sequential(*layers)

    def forward(self, prefix: EncodedTrace, pad_mask: torch.Tensor) -> Gaussian:
        """
        Args:
            prefix: The encoded prefix, read for both its summary and its events.
            pad_mask: True where a prefix position holds padding, `[batch_size, seq_len]`.
        Returns:
            `p(z | prefix)`.
        """
        pooled = self.pooling(source=prefix.events, pad_mask=pad_mask)  # [batch_size, pooled_dim]
        read = torch.cat(
            tensors=(prefix.summary, pooled), dim=-1
        )  # [batch_size, d_model + pooled_dim]
        return Gaussian.from_head(self.net(read))  # [batch_size, latent_dim] per field


class PosteriorNetwork(nn.Module):
    """`q(z | prefix, suffix)`: the distribution the latent is drawn from during training only.

    It hands the decoder a latent that already describes the suffix to reconstruct, which is
    what makes reconstruction learnable. At inference the suffix is unknown and the prior
    takes its place; the KL term trains the two to agree, which is what makes that
    substitution legitimate.

    Its pooling of the suffix is conditioned on the prefix summary, so it reads the suffix for
    what the prefix leaves undetermined rather than for the suffix in general, which is the
    quantity z is meant to carry.
    """

    def __init__(self, pooling_config: PoolingConfig, latent_config: LatentConfig, *, d_model: int):
        super().__init__()
        self.pooling = AttentionPooling(config=pooling_config, d_model=d_model)
        # Both summaries come out of the same encoder, so a single linear layer is enough here;
        # like the prior's output layer it emits mean and log-variance together.
        self.head = nn.Linear(
            in_features=2 * d_model + self.pooling.pooled_dim,
            out_features=2 * latent_config.latent_dim,
        )

    def forward(
        self, prefix_summary: torch.Tensor, suffix: EncodedTrace, pad_mask: torch.Tensor
    ) -> Gaussian:
        """
        Args:
            prefix_summary: The encoder's summary of the prefix, `[batch_size, d_model]`.
            suffix: The encoded ground-truth suffix, read for both its summary and its events.
            pad_mask: True where a suffix position holds padding, `[batch_size, seq_len]`.
        Returns:
            `q(z | prefix, suffix)`.
        """
        pooled = self.pooling(
            source=suffix.events, pad_mask=pad_mask, conditioning=prefix_summary
        )  # [batch_size, pooled_dim]
        read = torch.cat(
            tensors=(suffix.summary, prefix_summary, pooled), dim=-1
        )  # [batch_size, 2 * d_model + pooled_dim]
        return Gaussian.from_head(self.head(read))  # [batch_size, latent_dim] per field
