import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from src import paths
from src.configs import DataConfig, DeclareConfig, load_config
from src.datasets.codec import DatasetCodec
from src.logs.declare import discover_declare_model
from src.logs.filters import sort_log
from src.logs.io import read_log, write_log
from src.logs.keys import (
    ACTIVITY_KEY,
    CASE_ELAPSED_KEY,
    CASE_KEY,
    DAY_IN_WEEK_KEY,
    EVENT_DELTA_KEY,
    MIN_PREFIX_KEY,
    MISSING_FEATURE,
    REMAINING_TIME_KEY,
    RESOURCE_KEY,
    SECONDS_IN_DAY_KEY,
    TIMESTAMP_KEY,
)
from src.logs.split import out_of_time_split
from src.logs.timestamps import (
    add_calendar,
    add_case_elapsed,
    add_event_delta,
    add_remaining_time,
)
from src.paths import Split


def case_length_cutoff(log: pd.DataFrame, *, data_config: DataConfig) -> int:
    """Find the cutoff in events for dropping cases too long to fit the model's sequence tensors.

    Args:
        log: The whole preprocessed log, one row per event, before it is split.
        data_config: The `data` section, which either states the cutoff outright or names the
            percentile of case length to read it off at.
    Returns:
        The cutoff in events.
    """
    lengths = log.groupby(CASE_KEY)[CASE_KEY].size()
    return int(np.ceil(np.percentile(lengths, data_config.max_seq_len_percentile)))


def add_prefix_bounds(log: pd.DataFrame) -> pd.DataFrame:
    """Add the lower bound of the cut points every case may be split at, at its full width.
    Initialized to 1, updated to the actual lower bound after the out-of-time split.
    Args:
        log: The preprocessed log, one row per event.
    Returns:
        A copy of `log` with `MIN_PREFIX_KEY` added, constant over each case's rows.
    """
    log = log.copy()
    log[MIN_PREFIX_KEY] = 1
    return log


def preprocess(log: pd.DataFrame, *, feature_columns: list[str]) -> pd.DataFrame:
    """Preprocess an event log for model training.
    Args:
        log: Event log with columns already renamed to canonical names
            (see `src.logs.io.read_log`).
        feature_columns: `data.event_features`. The non-numeric ones are the categorical
            channels, and their gaps become `MISSING_FEATURE` here, so a value the log does not
            have is a value of its own from this point on rather than something every reader has
            to fill in again. The numeric ones keep their gaps: a missing number is carried by
            the present flag instead.
    Returns:
        A copy of `log` with the two timestamp proxies, the remaining time and the two calendar
        columns added, and its categorical feature columns filled.
    """
    log = add_event_delta(
        log,
        case_key=CASE_KEY,
        timestamp_key=TIMESTAMP_KEY,
        out_key=EVENT_DELTA_KEY,
    )
    log = add_case_elapsed(
        log,
        case_key=CASE_KEY,
        timestamp_key=TIMESTAMP_KEY,
        out_key=CASE_ELAPSED_KEY,
    )
    log = add_remaining_time(
        log,
        case_key=CASE_KEY,
        timestamp_key=TIMESTAMP_KEY,
        out_key=REMAINING_TIME_KEY,
    )
    log = add_calendar(
        log,
        timestamp_key=TIMESTAMP_KEY,
        day_key=DAY_IN_WEEK_KEY,
        seconds_key=SECONDS_IN_DAY_KEY,
    )
    # Filling leaves these columns as strings, so the same test in `DatasetCodec.fit`
    # sorts them into the same channels it would have before.
    for column in feature_columns:
        if not is_numeric_dtype(log[column]):
            log[column] = log[column].fillna(MISSING_FEATURE).astype(str)
    return log


def run(data_config: DataConfig, declare_config: DeclareConfig) -> None:
    """
    Preprocess and split a dataset, writing outputs next to the input.

    Reads `data/<dataset>/original.csv`, renames its structural columns to the canonical names
    used throughout the codebase, sorts it, extracts the temporal features, splits it into
    train/val/test out of time, and drops the cases too long to fit the model's sequence tensors.

    The vocabularies and normalization statistics the model is built against are fit here too,
    on the train split alone, and written beside it as `dataset.json`. The declarative model is
    discovered from the same split and written to `data/<dataset>/declare/model.decl`.

    Args:
        data_config: The `data` section of this dataset's experiment config.
        declare_config: The `declare` section, driving the discovery of the declarative model.
    """
    column_mapping = {
        data_config.case_key: CASE_KEY,
        data_config.activity_key: ACTIVITY_KEY,
        data_config.resource_key: RESOURCE_KEY,
        data_config.timestamp_key: TIMESTAMP_KEY,
    }
    dataset = data_config.name

    # Read the raw log.
    print(f'Preprocessing "{dataset}"...')
    log = read_log(
        paths.original_log(dataset),
        column_mapping=column_mapping,
        dtype=dict.fromkeys(data_config.string_features, str),
    )

    # Derive the columns the model reads, all of which are read off neighbouring rows or off the
    # size of a case, so the log has to be in order first.
    log = sort_log(log, case_key=CASE_KEY, timestamp_key=TIMESTAMP_KEY)
    log = preprocess(log, feature_columns=data_config.event_features)
    log = add_prefix_bounds(log)

    # Split the log into train/val/test out of time, dropping the cases longer than
    # `max_seq_len` at the point in the procedure where that does not bias its blocks.
    max_seq_len = case_length_cutoff(log, data_config=data_config)
    print(f'Splitting "{dataset}" into train/val/test...')
    train, val, test = out_of_time_split(
        log,
        case_key=CASE_KEY,
        timestamp_key=TIMESTAMP_KEY,
        val_frac=data_config.val_split,
        test_frac=data_config.test_split,
        max_seq_len=max_seq_len,
    )

    cases_read = log[CASE_KEY].nunique()
    dropped = cases_read - sum(rows[CASE_KEY].nunique() for rows in (train, val, test))
    print(
        f'Dropped {dropped} of {cases_read} cases longer than {max_seq_len} events '
        f'({dropped / cases_read:.2%})'
    )

    for split, rows in ((Split.TRAIN, train), (Split.VAL, val), (Split.TEST, test)):
        write_log(rows, paths.split_path(dataset=dataset, split=split))

    # Fit the vocabularies and normalization statistics on the train split, writng them out to
    # `dataset.json`. The generated values can be decoded back using the same codec.
    codec = DatasetCodec.fit(train, data_config=data_config, max_trace_length=max_seq_len)
    codec.save()

    # Discover the declarative model from the train split and write it out to `model.decl`.
    num_constraints = discover_declare_model(
        train,
        dataset=dataset,
        declare_config=declare_config,
    )

    print(
        f'Preprocessed "{dataset}": {len(train)} train, {len(val)} val, {len(test)} test '
        f'events, {len(codec.activity.vocab)} activities, '
        f'{len(codec.resource.vocab)} resources, '
        f'{len(codec.categorical_features)} categorical and '
        f'{len(codec.numeric_features)} numeric feature channels, '
        f'{num_constraints} declarative constraints'
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Turn a raw event log into the train/val/test CSVs the model consumes.'
    )
    parser.add_argument(
        '-c',
        '--config',
        type=Path,
        required=True,
        help="Path to this dataset's experiment config YAML.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run(data_config=config.data, declare_config=config.declare)


if __name__ == '__main__':
    main()
