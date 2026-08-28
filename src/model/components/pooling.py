import torch
from torch import nn

from src.configs.schema import PoolingConfig
from src.model.components.attention import MultiHeadAttention


class AttentionPooling(nn.Module):
    """Reads a padded sequence into a fixed number of rows, with learned queries of its own.

    One Perceiver-style resampler layer: `num_queries` queries cross-attend over every position of
    the sequence, then pass through a feedforward. Attention alone gives the queries access to
    every row; the feedforward is what leaves the pooled rows computed rather than merely read.

    A latent network holds one of these, so what it reads is pooled for its own purpose rather
    than for the encoder's, and its width is the queries' rather than the sequence's.
    """

    # Query rows start apart rather than at zero: identical rows attend identically and take
    # identical gradients, so a symmetric initialization would leave them tied for the whole run
    # and collapse the module onto one query.
    QUERY_INIT_STD = 0.02

    def __init__(self, config: PoolingConfig, *, d_model: int):
        super().__init__()
        self.queries = nn.Parameter(
            data=torch.empty(size=(1, config.num_queries, d_model))
        )  # [1, num_queries, d_model]
        nn.init.normal_(tensor=self.queries, std=self.QUERY_INIT_STD)

        self.attention = MultiHeadAttention(
            d_model=d_model, num_heads=config.num_heads, dropout=config.dropout
        )
        self.feedforward = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=config.feedforward_dim),
            nn.ReLU(),
            nn.Dropout(p=config.dropout),
            nn.Linear(in_features=config.feedforward_dim, out_features=d_model),
        )
        # Pre-norm, like the trace encoder's stack: each sublayer normalizes its input, and the
        # residual path is left clean.
        self.query_norm = nn.LayerNorm(normalized_shape=d_model)
        self.source_norm = nn.LayerNorm(normalized_shape=d_model)
        self.feedforward_norm = nn.LayerNorm(normalized_shape=d_model)
        # Pre-norm leaves the last residual stream unnormalized, and these rows are concatenated
        # with a summary that came out of the encoder's own closing norm, so the two arrive on one
        # scale only if this closes with a norm too.
        self.output_norm = nn.LayerNorm(normalized_shape=d_model)
        self.dropout = nn.Dropout(p=config.dropout)

        self.pooled_dim = config.num_queries * d_model

    def forward(
        self,
        source: torch.Tensor,
        pad_mask: torch.Tensor,
        conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pool a sequence into `num_queries` rows, flattened side by side.

        Args:
            source: The sequence to read, `[batch_size, seq_len, d_model]`.
            pad_mask: True where a position holds padding, `[batch_size, seq_len]`, from
                `Events.pad_mask`.
            conditioning: What to read the sequence for, `[batch_size, d_model]`, added to every
                query before the attention. None to pool the sequence on its own terms.
        Returns:
            `[batch_size, num_queries * d_model]`, each query keeping its own slot so the network
            reading this can tell them apart.
        """
        queries = self.queries.expand(
            source.size(dim=0), -1, -1
        )  # [batch_size, num_queries, d_model]
        if conditioning is not None:
            queries = queries + conditioning.unsqueeze(dim=1)  # [batch_size, num_queries, d_model]

        # Masked positions are dropped from every attention row, so padding contributes nothing
        # to any pooled row.
        keys_values = self.attention.project(self.source_norm(source))
        queries = queries + self.dropout(
            self.attention(
                query=self.query_norm(queries), keys_values=keys_values, key_padding_mask=pad_mask
            )
        )  # [batch_size, num_queries, d_model]
        queries = queries + self.dropout(
            self.feedforward(self.feedforward_norm(queries))
        )  # [batch_size, num_queries, d_model]
        return self.output_norm(queries).flatten(start_dim=1)  # [batch_size, num_queries*d_model]
