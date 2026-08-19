import pandas as pd


def sort_log(log: pd.DataFrame, *, case_key: str, timestamp_key: str) -> pd.DataFrame:
    """Order a log by case start time, and the events of a case by their own timestamp.

    Everything downstream reads a case as a run of consecutive rows in order: `add_event_delta`
    differences neighbouring rows, and the prefix bounds count events by their position. This is
    where that becomes true rather than assumed.

    Args:
        log: Event log, one row per event.
        case_key: Column identifying the case each event belongs to.
        timestamp_key: Column holding the (already parsed) event timestamp.
    Returns:
        A reindexed copy of `log`. The sort is stable, so events sharing a timestamp keep the
        order the log gave them, which is the order the prefix bounds are counted against.
    """
    # Rank the cases by start time and sort on the rank first, so a case's rows end up
    # consecutive and the cases end up in the order they began.
    case_start = log.groupby(case_key)[timestamp_key].min()
    rank = pd.Series(
        data=range(len(case_start)), index=case_start.sort_values(kind='mergesort').index
    )
    log = log.assign(_case_rank=log[case_key].map(rank))
    log = log.sort_values(by=['_case_rank', timestamp_key], kind='mergesort')
    return log.drop(columns='_case_rank').reset_index(drop=True)


def drop_cases_by_length(split: pd.DataFrame, *, case_key: str, max_seq_len: int) -> pd.DataFrame:
    """Drop the cases too long to fit the model's sequence tensors.

    Args:
        split: One split of the log, one row per event.
        case_key: Column identifying the case each event belongs to.
        max_seq_len: The longest case to keep, in events.
    Returns:
        A row-subset of `split` holding only the cases of at most `max_seq_len` events.
    """
    lengths = split.groupby(case_key)[case_key].transform('size')
    return split[lengths <= max_seq_len]


def case_durations(log: pd.DataFrame, *, case_key: str, timestamp_key: str) -> pd.Series:
    """Measure how long each case runs, first event to last.

    Args:
        log: Event log, one row per event.
        case_key: Column identifying the case each event belongs to.
        timestamp_key: Column holding the (already parsed) event timestamp.
    Returns:
        The duration of every case in days, indexed by case.
    """
    edges = log.groupby(case_key)[timestamp_key].agg(['min', 'max'])
    return (edges['max'] - edges['min']).dt.total_seconds() / 86_400.0


def drop_cases_by_duration(
    log: pd.DataFrame, *, case_key: str, timestamp_key: str, max_duration_days: float
) -> pd.DataFrame:
    """Drop the cases running far past the scale of the process they belong to.

    A case spanning decades is a broken timestamp rather than a slow process, and it is not merely
    an outlier the model sees a few of: every duration channel is standardized against statistics
    fit on the whole train split, so a handful of them set the scale the real inter-event deltas
    are then squeezed into.

    Args:
        log: Event log, one row per event.
        case_key: Column identifying the case each event belongs to.
        timestamp_key: Column holding the (already parsed) event timestamp.
        max_duration_days: The longest case to keep, first event to last, in days.
    Returns:
        A row-subset of `log` holding only the cases running at most `max_duration_days`.
    """
    durations = case_durations(log, case_key=case_key, timestamp_key=timestamp_key)
    return log[log[case_key].map(durations) <= max_duration_days]
