import numpy as np
import pandas as pd

from src.logs.filters import drop_long_cases
from src.logs.keys import MIN_PREFIX_KEY


def out_of_time_split(
    log: pd.DataFrame,
    *,
    case_key: str,
    timestamp_key: str,
    val_frac: float,
    test_frac: float,
    max_seq_len: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a log by time, withholding the prefixes that would leak across the boundary.

    The separation time is the start of the case that begins once `1 - test_frac` of them
    already have. Cutting there alone still leaks: a case running across that moment would be
    learned from before its later events could have been observed. Test holds every case still
    running at the separation, so a crossing case is a test case whose earlier prefixes a
    training run could already have seen; those prefixes are withheld from scoring rather than
    the case being dropped, which is what `MIN_PREFIX_KEY` records.

    Args:
        log: Event log, one row per event, sorted by case start and then by timestamp, carrying
            `MIN_PREFIX_KEY` at its full range.
        case_key: Column identifying the case each event belongs to.
        timestamp_key: Column holding the (already parsed) event timestamp.
        val_frac: Fraction of all cases assigned to the validation set, taken from the training
            cases as the latest-starting ones.
        test_frac: Fraction of all cases the separation time is placed at.
        max_seq_len: The longest case to keep, in events. Applied here rather than by the caller
            because the validation cases are carved out of what survives it.
    Returns:
        `(train, val, test)` DataFrames, each a row-subset of `log` with `MIN_PREFIX_KEY`
        narrowed on the crossing cases. Every case left is short enough already, so a further
        `drop_long_cases` would find nothing.
    """
    # Compute each case's start and stop timestamps
    edges = log.groupby(case_key)[timestamp_key].agg(['min', 'max'])
    starts_ts, stops_ts = edges['min'], edges['max']

    # Find the index of the first test case and its start time
    first_test_case = int(len(edges) * (1 - test_frac))
    separation_ts = np.sort(starts_ts.values)[first_test_case]

    # The training set keeps every case that has already finished before the separation
    train_cases = stops_ts.index[stops_ts.values < separation_ts]
    train = log[log[case_key].isin(train_cases)]

    # The test set keeps every case that was running at the separation.
    test_cases = stops_ts.index[stops_ts.values >= separation_ts]
    crossing = test_cases.difference(starts_ts.sort_values().index[first_test_case:])
    # All events of crossing cases before the cut point are removed from scoring by
    # narrowing their `MIN_PREFIX_KEY.
    test = _bound_crossing_prefixes(
        log[log[case_key].isin(test_cases)],
        case_key=case_key,
        timestamp_key=timestamp_key,
        separation_ts=separation_ts,
        crossing=crossing,
    )

    # Before the validation carve-out, so the cases it reads are the ones that survive: dropping
    # afterwards would leave validation holding a share of a set it was not taken from.
    train = drop_long_cases(train, case_key=case_key, max_seq_len=max_seq_len)
    test = drop_long_cases(test, case_key=case_key, max_seq_len=max_seq_len)

    # Validation is the tail of the training block by case start, so it stands out of time to
    # training the same way the test block stands out of time to both.
    val_share = val_frac / (1 - test_frac)
    train_starts = train.groupby(case_key)[timestamp_key].min().sort_values()
    val_cases = train_starts.index[int(len(train_starts) * (1 - val_share)) :]

    val = train[train[case_key].isin(val_cases)]
    train = train[~train[case_key].isin(val_cases)]

    return train, val, test


def _bound_crossing_prefixes(
    split: pd.DataFrame,
    *,
    case_key: str,
    timestamp_key: str,
    separation_ts: np.datetime64,
    crossing: pd.Index,
) -> pd.DataFrame:
    """Narrow the cut points of the cases running across the separation time.

    Args:
        split: The test split, holding the crossing cases, one row per event.
        case_key: Column identifying the case each event belongs to.
        timestamp_key: Column holding the (already parsed) event timestamp.
        separation_ts: The moment the split was cut at.
        crossing: The cases whose prefixes are only partly test's to score.
    Returns:
        A copy of `split` with `MIN_PREFIX_KEY` narrowed on those cases, to their first cut point
        whose prefix reaches past the separation.
    """
    split = split.copy()
    rows = split[split[case_key].isin(crossing)]
    # A cut point of k takes the first k events, so a bound is the 1-based position of the event
    # the prefix has to reach, one past that event's 0-based index in its case.
    positions = rows.groupby(case_key).cumcount() + 1

    # Strictly after the separation: an event landing exactly on it is one a training case could
    # still have been running through. Compared as numpy values, since a tz-aware column and a
    # numpy timestamp are not comparable as pandas objects.
    selected = rows[timestamp_key].values > separation_ts
    bounds = positions[selected].groupby(rows[case_key][selected]).min()

    # A case the bound says nothing about keeps the full range it was written with.
    split[MIN_PREFIX_KEY] = split[case_key].map(bounds).fillna(split[MIN_PREFIX_KEY]).astype(int)
    return split
