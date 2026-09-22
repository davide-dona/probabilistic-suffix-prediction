from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from omegaconf import DictConfig
from pandas.api.types import is_numeric_dtype
from pydantic import BaseModel, ConfigDict, Field

from src import paths
from src.datasets.codec.activity import ActivityCodec
from src.datasets.codec.categorical import (
    ACTIVITY_TOKENS,
    FEATURE_TOKENS,
    RESOURCE_TOKENS,
    CategoricalColumn,
)
from src.datasets.codec.categorical import (
    encode_features as encode_categorical_features,
)
from src.datasets.codec.continuous import NumericColumn
from src.datasets.codec.continuous import encode_features as encode_numeric_features
from src.logs import (
    ACTIVITY_KEY,
    CASE_KEY,
    INTER_EVENT_TIME_KEY,
    REMAINING_TIME_KEY,
    RESOURCE_KEY,
    Split,
    read_log,
)


class DatasetCodec(BaseModel):
    """What a dataset's values are encoded through, fit on the train split and written beside
    the splits."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    activity: CategoricalColumn
    resource: CategoricalColumn
    # inter_event_time is read by both the encoders and the decoder; remaining_time is
    # decoder-only.
    inter_event_time: NumericColumn
    remaining_time: NumericColumn

    # The columns `data.event_features` names, sorted by dtype into the two kinds.
    categorical_features: tuple[CategoricalColumn, ...]
    numeric_features: tuple[NumericColumn, ...]

    # The maximum trace length the splits were preprocessed to: `data.max_seq_len_percentile` of
    # case length, measured on the whole log. Written to `dataset.json`, since it is a property of
    # the fit rather than of the config, which only holds the percentile.
    max_trace_length: int

    # Which dataset the splits belong to. From the config rather than the fit, and excluded from
    # `dataset.json`, since it is the directory the file is written into.
    dataset: str = Field(..., exclude=True)

    @property
    def num_feature_categories(self) -> int:
        """Rows in the table every categorical feature channel shares: one PAD, then a block
        per feature. 1 on a dataset with no categorical features, where no table is built."""
        return 1 + sum(feature.num_rows for feature in self.categorical_features)

    @property
    def activity_codes(self) -> ActivityCodec:
        """A fresh codebook for compact activity-sequence storage and comparison.

        Its code order follows the activity channel's complete decode vocabulary, including the
        special tokens. Each caller receives a fresh codebook because `ActivityCodec` can extend
        itself for an activity outside that vocabulary.
        """
        return ActivityCodec.from_vocabulary(self.activity.names)

    @classmethod
    def fit(
        cls, train: pd.DataFrame, *, data_config: DictConfig, max_trace_length: int
    ) -> DatasetCodec:
        """Fit the codec on the train split.
        Args:
            train: The train split, as `pipelines/preprocess.py` holds it before writing.
            data_config: The `data` section, for the feature columns, which of them are
                log-scaled, and how the two time targets are scaled.
            max_trace_length: The sequence length splits were preprocessed to, from
                `pipelines.preprocess.case_length_cutoff`.
        Returns:
            The codec to write beside the splits.
        """
        categorical_features, numeric_features = _fit_event_features(
            train=train,
            columns=data_config.event_features,
            log_scaled=set(data_config.log_scaled_features),
        )
        return cls(
            activity=CategoricalColumn.fit(
                train, column=ACTIVITY_KEY, special_tokens=ACTIVITY_TOKENS
            ),
            resource=CategoricalColumn.fit(
                train, column=RESOURCE_KEY, special_tokens=RESOURCE_TOKENS
            ),
            inter_event_time=NumericColumn.fit(
                train, column=INTER_EVENT_TIME_KEY, log=data_config.log_scaled_inter_event_time
            ),
            remaining_time=NumericColumn.fit(
                train, column=REMAINING_TIME_KEY, log=data_config.log_scaled_remaining_time
            ),
            categorical_features=categorical_features,
            numeric_features=numeric_features,
            dataset=data_config.name,
            max_trace_length=max_trace_length,
        )

    @classmethod
    def load(cls, data_config: DictConfig) -> DatasetCodec:
        """Load the codec previously fit for a dataset.

        The one place a config becomes a codec: what comes back names the dataset, so everything
        read from disk afterwards is asked of the codec alone.

        Args:
            data_config: The `data` section, naming the dataset.
        Returns:
            The dataset codec, with `dataset` taken from the config and everything else,
            `max_trace_length` included, read back from what preprocessing fit.
        Raises:
            FileNotFoundError: If the dataset has not been preprocessed.
        """
        name = data_config.name
        path = paths.CODEC.require(name)
        payload = json.loads(path.read_text())
        return cls.model_validate(payload | {'dataset': name})

    def save(self) -> Path:
        """Write this codec to its own directory, and return where it went."""
        path = paths.CODEC.prepare(self.dataset)
        path.write_text(self.model_dump_json(indent=2))
        return path

    def split_path(self, split: Split) -> Path:
        """Where one preprocessed split of this dataset is kept.

        Args:
            split: Which of the three to name.
        Returns:
            The path to that split's file.
        """
        return paths.PROCESSED_SPLIT.path(dataset=self.dataset, split=split)

    def read_split(self, split: Split) -> pd.DataFrame:
        """Read a preprocessed split with categorical columns forced to text."""
        categorical = (self.activity, self.resource, *self.categorical_features)
        text_columns = {CASE_KEY: str} | {column.column: str for column in categorical}
        return read_log(self.split_path(split), dtype=text_columns)

    def encode_categorical_features(self, log: pd.DataFrame) -> torch.Tensor:
        """Pack the categorical features into shared-table indices."""
        return encode_categorical_features(self.categorical_features, log)

    def encode_numeric_features(self, log: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack the numeric feature values and their presence flags."""
        return encode_numeric_features(self.numeric_features, log)


def _fit_event_features(
    *,
    train: pd.DataFrame,
    columns: list[str],
    log_scaled: set[str],
) -> tuple[tuple[CategoricalColumn, ...], tuple[NumericColumn, ...]]:
    """Sort the configured columns into categorical and numeric, and fit each on the train split.
    Args:
        train: The split every vocabulary and statistic is fit on.
        columns: The configured columns, in the order they will occupy the shared table and the
            embedding projection's input.
        log_scaled: Which of them take a log1p before the standardization. A name in here that
            turns out categorical simply never comes up, the column having nothing to scale.
    Returns:
        The categorical features with their table offsets assigned, and the numeric features
        with their fitted statistics.
    """
    categorical: list[CategoricalColumn] = []
    numeric: list[NumericColumn] = []
    # Row 0 of the shared table is the PAD every feature uses, so the first block starts at 1.
    offset = 1

    for column in columns:
        if is_numeric_dtype(train[column]):
            numeric.append(NumericColumn.fit(train, column=column, log=column in log_scaled))
        else:
            feature = CategoricalColumn.fit(
                train, column=column, special_tokens=FEATURE_TOKENS, offset=offset
            )
            categorical.append(feature)
            # The next feature's block starts after this one's vocabulary and its UNK row.
            offset += feature.num_rows

    return tuple(categorical), tuple(numeric)
