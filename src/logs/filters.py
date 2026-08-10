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


def drop_long_cases(split: pd.DataFrame, *, case_key: str, max_seq_len: int) -> pd.DataFrame:
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
