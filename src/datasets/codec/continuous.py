from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict


class NumericColumn(BaseModel):
    """One numeric channel of the log, and the train-split statistics it is standardized with.
    - log: Whether the values pass through a log1p before the standardization.
    - mean: The mean of the train values, after the log1p if there is one.
    - std: The standard deviation of the same, 1.0 for a channel that never varies.
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    column: str
    log: bool
    mean: float
    std: float

    @classmethod
    def fit(cls, train: pd.DataFrame, *, column: str, log: bool) -> NumericColumn:
        """Fit one channel's standardization on the train split.

        Missing values take no part in the fit, so a channel with gaps is standardized on the
        values it does have.

        Args:
            train: The split every statistic is fit on, and the only one any is ever fit on.
            column: Which column of it this channel reads.
            log: Whether to put the values through a log1p before taking mean and deviation.
        Returns:
            The channel, holding the statistics its values are standardized with.
        Raises:
            ValueError: If the column holds no finite value, or a negative one on a log-scaled
                channel, where log1p is undefined below -1 and compresses the rest towards it.
        """
        values = train[column].to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError(f'column "{column}" holds no finite value on the train split')
        if log and finite.min() < 0:
            raise ValueError(
                f'column "{column}" is log-scaled but holds negative values '
                f'(min {finite.min()}), which log1p cannot represent'
            )

        scaled = cls._scale(finite, log=log)
        # A channel whose train values are all identical has no deviation to divide by. 1.0
        # leaves every one of them standardizing to 0.0 and denormalizing back to the mean.
        std = float(scaled.std())
        return cls(
            column=column,
            log=log,
            mean=float(scaled.mean()),
            std=std if std > 0 else 1.0,
        )

    @staticmethod
    def _scale(values: np.ndarray, *, log: bool) -> np.ndarray:
        """Apply this channel's transform to the raw values."""
        return np.log1p(values) if log else values

    def normalize(self, values: np.ndarray) -> np.ndarray:
        """Standardize raw values against the train split's mean and deviation.

        Args:
            values: The raw values to normalize.
        Returns:
            The same values standardized, as float32. Nothing bounds them: a val/test value
            beyond anything the train split held keeps its distance rather than being pulled
            back to a range.
        """
        return ((self._scale(values, log=self.log) - self.mean) / self.std).astype(np.float32)

    def denormalize(self, normalized: np.ndarray) -> np.ndarray:
        """Read standardized values back as the raw quantity they came from, exactly: the
        transform loses nothing on the way in."""
        scaled = np.asarray(normalized, dtype=np.float64) * self.std + self.mean
        return np.expm1(scaled) if self.log else scaled

    def encode(self, log: pd.DataFrame) -> np.ndarray:
        """Standardize this channel's raw values, with a missing one pinned to exactly 0.0.

        Args:
            log: The events as a preprocessed split holds them.
        Returns:
            `[len(log)]` of float32.
        """
        raw = log[self.column].to_numpy(dtype=np.float64)
        # The fit skipped the gaps, so a gap is zeroed rather than standardized: multiplying by
        # the flag is what pins it to exactly 0.0, which is the train mean of the channel.
        finite = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        return self.normalize(finite) * self.present(log)

    def present(self, log: pd.DataFrame) -> np.ndarray:
        """Mark the rows this channel has a value for.

        Args:
            log: The events as a preprocessed split holds them.
        Returns:
            `[len(log)]` of float32, 0.0 where the log had no value. 0.0 is also a legitimate
            standardized value, which is what this flag is for.
        """
        return np.isfinite(log[self.column].to_numpy(dtype=np.float64)).astype(np.float32)


def encode_features(
    features: tuple[NumericColumn, ...], log: pd.DataFrame
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack numeric feature values and presence flags, each of shape
    `[len(log), num_numeric]`."""
    if not features:
        empty = torch.zeros(size=(len(log), 0), dtype=torch.float32)
        return empty, empty.clone()
    return (
        torch.stack(tensors=[torch.from_numpy(feature.encode(log)) for feature in features], dim=1),
        torch.stack(
            tensors=[torch.from_numpy(feature.present(log)) for feature in features], dim=1
        ),
    )
