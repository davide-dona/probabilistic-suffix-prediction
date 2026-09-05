import argparse
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

from src import paths
from src.cli import banner
from src.configs import load_dataset_config
from src.logs.filters import case_durations
from src.logs.io import read_log
from src.logs.keys import (
    ACTIVITY_KEY,
    CASE_ELAPSED_KEY,
    CASE_KEY,
    DAY_COS_KEY,
    DAY_SIN_KEY,
    EVENT_DELTA_KEY,
    MIN_PREFIX_KEY,
    MISSING_FEATURE,
    REMAINING_TIME_KEY,
    SECONDS_COS_KEY,
    SECONDS_SIN_KEY,
    TIMESTAMP_KEY,
    Split,
)

# Columns preprocessing derives from the timestamp rather than the raw dataset carrying them.
_DERIVED_COLUMNS = {
    EVENT_DELTA_KEY,
    CASE_ELAPSED_KEY,
    REMAINING_TIME_KEY,
    MIN_PREFIX_KEY,
    DAY_SIN_KEY,
    DAY_COS_KEY,
    SECONDS_SIN_KEY,
    SECONDS_COS_KEY,
}


def read_processed_log(dataset: str) -> pd.DataFrame:
    """Read the full processed log a dataset's splits partition, before the out-of-time split.

    Concatenates the three splits rather than rereading and reprocessing the raw log: splitting
    only partitions cases by time and narrows `MIN_PREFIX_KEY` on the ones crossing the
    boundary, so no row is added or dropped by it.

    Args:
        dataset: The dataset whose splits to read, from where preprocessing wrote them.
    Returns:
        The processed log, one row per event.
    """
    splits = [
        read_log(paths.PROCESSED_SPLIT.require(dataset=dataset, split=split)) for split in Split
    ]
    return pd.concat(splits, ignore_index=True)


def summarize_feature(log: pd.DataFrame, column: str) -> dict[str, object]:
    """Describe one event feature: whether it varies within a case, its type, and its spread.

    Args:
        log: The processed log, one row per event.
        column: The feature column to describe.
    Returns:
        One row of the feature table.
    """
    values = log[column]
    level = 'case' if log.groupby(CASE_KEY)[column].nunique(dropna=False).le(1).all() else 'event'

    if is_numeric_dtype(values):
        return {
            'feature': column,
            'level': level,
            'type': 'numeric',
            'missing': f'{values.isna().mean():.1%}',
            'mean': f'{values.mean():.3g}',
            'std': f'{values.std():.3g}',
            'categories': '',
        }
    return {
        'feature': column,
        'level': level,
        'type': 'categorical',
        'missing': f'{(values == MISSING_FEATURE).mean():.1%}',
        'mean': '',
        'std': '',
        'categories': int(values.nunique()),
    }


def summarize_dataset(dataset: str, *, event_features: list[str]) -> None:
    """Print the case-, event- and activity-level statistics of one dataset's processed log.

    Args:
        dataset: The dataset to summarize.
        event_features: `data.event_features` from its config.
    """
    log = read_processed_log(dataset)

    lengths = log.groupby(CASE_KEY).size()
    durations = case_durations(log, case_key=CASE_KEY, timestamp_key=TIMESTAMP_KEY)
    variants = log.groupby(CASE_KEY)[ACTIVITY_KEY].agg(tuple).nunique()

    banner(
        f'"{dataset}"',
        {
            'cases': f'{lengths.size:,}',
            'events': f'{len(log):,}',
            'variants': f'{variants:,}',
            'activities': f'{log[ACTIVITY_KEY].nunique():,}',
            'case length': f'{lengths.mean():.1f} +/- {lengths.std():.1f} events',
            'case duration': f'{durations.mean():.2f} +/- {durations.std():.2f} days',
        },
    )

    features = [column for column in event_features if column not in _DERIVED_COLUMNS]
    if features:
        table = pd.DataFrame(summarize_feature(log, column) for column in features)
        print(table.to_string(index=False))
    print(flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Print per-dataset statistics off the processed log(s), before the split.'
    )
    parser.add_argument(
        '-c',
        '--configs',
        nargs='*',
        type=paths.existing_file,
        metavar='CONFIG',
        help='Dataset configs to summarize, e.g. config/datasets/bpic17.yaml. Defaults to every '
        'config under config/datasets/.',
    )
    args = parser.parse_args()
    configs = args.configs or sorted(Path('config/datasets').glob('*.yaml'))

    for config_path in configs:
        data_config = load_dataset_config(config_path).data
        summarize_dataset(data_config.name, event_features=data_config.event_features)


if __name__ == '__main__':
    main()
