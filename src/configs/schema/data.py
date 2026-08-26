from __future__ import annotations

from pydantic import Field, model_validator

from src.logs.keys import EVENT_DELTA_KEY

from .base import NAME_PATTERN, StrictModel


class DataConfig(StrictModel):
    """How the raw log is read, preprocessed and split into train/val/test."""

    name: str = Field(
        ...,
        pattern=NAME_PATTERN,
        description='The log this config describes, e.g. `sepsis`. Names its preprocessed '
        'splits, codec, declarative model and figures',
    )

    case_key: str = Field(..., description='Raw column identifying the case each event belongs to')
    activity_key: str = Field(..., description='Raw column identifying the activity label')
    resource_key: str = Field(..., description='Raw column identifying the resource')
    timestamp_key: str = Field(..., description='Raw column holding the event timestamp')

    train_split: float = Field(..., gt=0.0, lt=1.0)
    val_split: float = Field(..., gt=0.0, lt=1.0)
    test_split: float = Field(..., gt=0.0, lt=1.0)

    max_seq_len_percentile: float = Field(
        ...,
        gt=0.0,
        le=100.0,
        description='Cases longer than this percentile of case length are dropped at '
        'preprocessing time. The cutoff is what sequence tensors are padded to',
    )

    max_case_duration_percentile: float = Field(
        ...,
        gt=0.0,
        le=100.0,
        description='Cases running longer than this percentile of case duration are dropped at '
        'preprocessing time, before the log is split. A handful of broken timestamps otherwise '
        'set the statistics every duration channel is standardized against',
    )

    string_features: list[str] = Field(
        ...,
        description='Columns read as strings rather than by whatever dtype pandas infers, for '
        'the identifiers and flags that would otherwise become numeric channels',
    )

    event_features: list[str] = Field(
        ...,
        description='Every per-event column the encoders read beside the activity and the '
        'resource. Names either a raw column of the log or one preprocessing derives from the '
        'timestamp (`ts_start`, `day_sin`, `day_cos`, `seconds_sin`, `seconds_cos`; see '
        '`src/logs/keys.py`). A numeric one becomes a value and a present flag, anything '
        'else a vocabulary. Changing this list requires re-preprocessing the dataset',
    )

    log_scaled_features: list[str] = Field(
        ...,
        description='Which `event_features` take a log1p before being standardized, for columns '
        'spanning several orders of magnitude. Empty matches the baselines, which standardize '
        'every numeric column raw',
    )

    log_scaled_remaining_time: bool = Field(
        ...,
        description='Whether the remaining-time target takes a log1p before being standardized. '
        'Off matches the baselines, which standardize durations raw. `ts_start` is scaled '
        'through `log_scaled_features` instead',
    )

    log_scaled_time_to_next: bool = Field(
        ...,
        description='Whether the time-to-next-event target takes a log1p before being '
        'standardized. It is the more skewed of the two, a few long waits behind a mass of '
        'same-day events',
    )

    @model_validator(mode='after')
    def _splits_sum_to_one(self) -> DataConfig:
        total = self.train_split + self.val_split + self.test_split
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f'train/val/test splits must sum to 1.0, got {total}')
        return self

    @model_validator(mode='after')
    def _log_scaled_features_are_event_features(self) -> DataConfig:
        # A column named here but not read as a feature would scale nothing, silently.
        unknown = sorted(set(self.log_scaled_features) - set(self.event_features))
        if unknown:
            raise ValueError(
                f'log_scaled_features names columns that are not event_features: '
                f'{", ".join(unknown)}'
            )
        return self

    @model_validator(mode='after')
    def _event_delta_is_not_an_event_feature(self) -> DataConfig:
        # ts_prev would otherwise be fit twice: once here, once as time_to_next.
        if EVENT_DELTA_KEY in self.event_features:
            raise ValueError(
                f'event_features names "{EVENT_DELTA_KEY}", which is read through '
                f'DatasetCodec.time_to_next instead and must not also be listed here'
            )
        return self


class DeclareConfig(StrictModel):
    """Declarative-model discovery, run once per dataset at preprocessing time on the train
    split only.
    """

    consider_vacuity: bool = Field(
        ..., description='Whether a trace that never activates a constraint counts as satisfying it'
    )
    min_support: float = Field(
        ...,
        gt=0.0,
        le=1.0,
        description='Fraction of train traces a constraint must hold on to be kept',
    )
    itemsets_support: float = Field(
        ...,
        gt=0.0,
        le=1.0,
        description='Support floor for the frequent activity itemsets candidate constraints '
        'are built from',
    )
    max_cardinality: int = Field(
        ..., gt=0, description='Highest n tried for the templates that take one (`Existence3[A]`)'
    )
