import argparse

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from src import paths
from src.configs import DataConfig, DeclareConfig, load_dataset_config
from src.datasets.codec import DatasetCodec
from src.logs.declare import discover_declare_model
from src.logs.filters import sort_log
from src.logs.io import read_original_log, write_log
from src.logs.keys import (
    CASE_ELAPSED_KEY,
    CASE_KEY,
    DAY_IN_WEEK_KEY,
    EVENT_DELTA_KEY,
    MIN_PREFIX_KEY,
    MISSING_FEATURE,
    REMAINING_TIME_KEY,
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


def run(data_config: DataConfig, declare_config: DeclareConfig, *, skip_discovery: bool) -> None:
    """
    Preprocess and split a dataset, writing outputs next to the input.

    Reads `data/<dataset>/original.csv`, renames its structural columns to the canonical names
    used throughout the codebase, sorts it, extracts the temporal features, splits it into
    train/val/test out of time, and drops the cases too long to fit the model's sequence tensors.

    The vocabularies and normalization statistics the model is built against are fit here too,
    on the train split alone, and written beside it as `dataset.json`. The declarative model is
    discovered from the same split and written to `data/<dataset>/declare/model.decl`, unless
    `skip_discovery` is set, since neither training nor generation reads it and discovery is the
    slowest step here.

    Args:
        data_config: The `data` section of this dataset's experiment config.
        declare_config: The `declare` section, driving the discovery of the declarative model.
        skip_discovery: Whether to skip discovering the declarative model. Evaluation will fail
            until preprocessing is rerun without this flag.
    """
    dataset = data_config.name

    # Read the raw log.
    print(f'Preprocessing "{dataset}"...', flush=True)
    print('Reading the original log...', flush=True)
    log = read_original_log(data_config)

    # Derive the columns the model reads, all of which are read off neighbouring rows or off the
    # size of a case, so the log has to be in order first.
    print('Deriving the temporal and calendar features...', flush=True)
    log = sort_log(log, case_key=CASE_KEY, timestamp_key=TIMESTAMP_KEY)
    log = preprocess(log, feature_columns=data_config.event_features)
    log = add_prefix_bounds(log)

    # Split the log into train/val/test out of time, dropping the cases longer than
    # `max_seq_len` at the point in the procedure where that does not bias its blocks.
    max_seq_len = case_length_cutoff(log, data_config=data_config)
    print(f'Splitting "{dataset}" into train/val/test...', flush=True)
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
        f'({dropped / cases_read:.2%})',
        flush=True,
    )

    print('Writing the splits...', flush=True)
    for split, rows in ((Split.TRAIN, train), (Split.VAL, val), (Split.TEST, test)):
        write_log(rows, paths.split_path(dataset=dataset, split=split))

    # Fit the vocabularies and normalization statistics on the train split, writng them out to
    # `dataset.json`. The generated values can be decoded back using the same codec.
    print('Fitting the dataset codec...', flush=True)
    codec = DatasetCodec.fit(train, data_config=data_config, max_trace_length=max_seq_len)
    codec.save()

    # Discover the declarative model from the train split and write it out to `model.decl`,
    # unless discovery was skipped.
    if skip_discovery:
        constraints_summary = 'declarative model discovery skipped'
    else:
        # The slowest step of the pipeline by a wide margin.
        print('Discovering the declarative model...', flush=True)
        num_constraints = discover_declare_model(
            train,
            dataset=dataset,
            declare_config=declare_config,
        )
        constraints_summary = f'{num_constraints} declarative constraints'

    print(
        f'Preprocessed "{dataset}": {len(train)} train, {len(val)} val, {len(test)} test '
        f'events, {len(codec.activity.vocab)} activities, '
        f'{len(codec.resource.vocab)} resources, '
        f'{len(codec.categorical_features)} categorical and '
        f'{len(codec.numeric_features)} numeric feature channels, '
        f'{constraints_summary}',
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Turn a raw event log into the train/val/test CSVs the model consumes.'
    )
    parser.add_argument(
        '-c',
        '--config',
        required=True,
        help="Name of this dataset's experiment config, from config/datasets/ (e.g. 'bpic17').",
    )
    parser.add_argument(
        '--skip-discovery',
        action='store_true',
        help='Skip discovering the declarative model, the slowest preprocessing step and one '
        'neither training nor generation reads. Evaluation needs it, so rerun without this '
        'flag before evaluating.',
    )
    args = parser.parse_args()

    config = load_dataset_config(args.config)
    run(data_config=config.data, declare_config=config.declare, skip_discovery=args.skip_discovery)


if __name__ == '__main__':
    main()
