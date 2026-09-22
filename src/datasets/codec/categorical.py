from __future__ import annotations

from functools import cached_property

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict

from src.logs import EOT_TOKEN, PAD_TOKEN, SOS_TOKEN, UNK_TOKEN

# The special tokens every channel carries, in the order they are indexed.
ACTIVITY_TOKENS = (PAD_TOKEN, EOT_TOKEN, SOS_TOKEN, UNK_TOKEN)
RESOURCE_TOKENS = (PAD_TOKEN, EOT_TOKEN, UNK_TOKEN)
FEATURE_TOKENS = (UNK_TOKEN,)


class CategoricalColumn(BaseModel):
    """One categorical channel of the log, and the vocabulary it is embedded through."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    column: str
    vocab: tuple[str, ...]
    special_tokens: tuple[str, ...]
    offset: int

    @classmethod
    def fit(
        cls,
        train: pd.DataFrame,
        *,
        column: str,
        special_tokens: tuple[str, ...],
        offset: int = 0,
    ) -> CategoricalColumn:
        """Fit one channel's vocabulary on the train split.

        Args:
            train: The split every vocabulary is fit on.
            column: Which column of it this channel reads.
            special_tokens: The tokens following the vocabulary, in the order they are indexed.
            offset: Where this channel's block starts in the table it is embedded with.
        Returns:
            The channel, its every index derivable from the vocabulary it holds.
        """
        return cls(
            column=column,
            vocab=tuple(sorted(train[column].astype(str).unique().tolist())),
            special_tokens=special_tokens,
            offset=offset,
        )

    @property
    def num_rows(self) -> int:
        """Rows this channel owns in its table, its special tokens included."""
        return len(self.vocab) + len(self.special_tokens)

    @cached_property
    def to_index(self) -> dict[str, int]:
        """Each value of the train split to its row, the vocabulary following the special tokens.
        The special tokens are absent: no raw value maps to one, which is what makes an unseen
        value fall through to `unk_index`."""
        start = self.offset + len(self.special_tokens)
        return {value: start + i for i, value in enumerate(self.vocab)}

    @cached_property
    def from_index(self) -> dict[int, str]:
        """Each row back to the value it stands for, the special tokens included.

        UNK is one-way in spirit - many raw values map to it - but it still appears here, since
        reading a prediction back has to name the row somehow, and which value it was is exactly
        what the encoding threw away.
        """
        return {i: value for value, i in self.to_index.items()} | {
            self._index_of(token): token for token in self.special_tokens
        }

    @property
    def names(self) -> tuple[str, ...]:
        """Every value `decode` can name, in row order, the special tokens included.

        For the activity channel, `DatasetCodec.activity_codes` uses this order to give every
        decoded value a stable compact code.
        """
        return tuple(self.from_index[row] for row in sorted(self.from_index))

    @property
    def eot_index(self) -> int:
        """The row marking the end of a trace."""
        return self._index_of(EOT_TOKEN)

    @property
    def pad_index(self) -> int:
        """The row filling a sequence out to `max_trace_length`, row 0 of any channel
        carrying it."""
        return self._index_of(PAD_TOKEN)

    @property
    def sos_index(self) -> int:
        """The row a generation starts from."""
        return self._index_of(SOS_TOKEN)

    @property
    def unk_index(self) -> int:
        """Every value val/test holds that the train split did not, collapsed into one row."""
        return self._index_of(UNK_TOKEN)

    def encode(self, log: pd.DataFrame) -> np.ndarray:
        """Map this channel's raw values to rows of the table it is embedded with, with an UNK
        for values the train split did not see.

        Args:
            log: The events as a preprocessed split holds them.
        Returns:
            `[len(log)]` of int64 rows.
        """
        return log[self.column].map(self.to_index).fillna(self.unk_index).to_numpy(dtype=np.int64)

    def decode(self, indices: np.ndarray, *, length: int) -> list[str]:
        """Read a run of this channel's indices back into the log's own values.

        Args:
            indices: The channel's indices for one sequence, int64, `[seq_len]`.
            length: How many of them to keep, the rest being padding.
        Returns:
            The values, in order.
        """
        return [self.from_index[int(index)] for index in indices[:length]]

    def _index_of(self, token: str) -> int:
        """The row of one special token.

        Raises:
            ValueError: If this channel does not carry that token - a resource is never generated,
                so it has no SOS, and asking for one is a bug rather than a missing default.
        """
        if token not in self.special_tokens:
            raise ValueError(f'column "{self.column}" carries no {token} token')
        return self.offset + self.special_tokens.index(token)


def encode_features(features: tuple[CategoricalColumn, ...], log: pd.DataFrame) -> torch.Tensor:
    """Pack categorical feature channels into shared-table indices of shape
    `[len(log), num_categorical]`."""
    if not features:
        return torch.zeros(size=(len(log), 0), dtype=torch.long)
    return torch.stack(
        tensors=[torch.tensor(data=feature.encode(log), dtype=torch.long) for feature in features],
        dim=1,
    )
