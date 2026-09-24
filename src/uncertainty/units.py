from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import read_prefix_scores, require_columns

PREFIX_KEYS = ('case_id', 'prefix_len')


def _aligned(
    dataset: str, files: dict[str, Path], metrics: Sequence[str]
) -> tuple[np.ndarray, pd.MultiIndex]:
    """Align model scores on identical sorted prefix keys."""
    frames = []
    prefixes = None
    for file in files.values():
        columns = (*PREFIX_KEYS, *metrics)
        require_columns(path=file, columns=columns)
        frame = read_prefix_scores(path=file, columns=columns)
        if frame[list(PREFIX_KEYS)].isna().any().any():
            raise ValueError(f'{file} contains missing prefix keys.')
        frame = frame.set_index(list(PREFIX_KEYS)).sort_index()
        if frame.index.has_duplicates:
            raise ValueError(f'{file} scores the same prefix twice, so it cannot be compared.')
        if prefixes is not None and not frame.index.equals(prefixes):
            raise ValueError(f'The runs of {dataset} do not score the same prefixes.')
        prefixes = frame.index
        frames.append(frame[list(metrics)].to_numpy(dtype=np.float64))

    assert prefixes is not None
    return np.stack(frames, axis=1), prefixes


def by_case(
    dataset: str, files: dict[str, Path], metrics: Sequence[str]
) -> Iterator[tuple[list[str], np.ndarray, np.ndarray]]:
    """Yield case totals and prefix counts for each reporting population.

    Args:
        dataset: Dataset name used in validation errors.
        files: Per-prefix score files in model order.
        metrics: Metrics to aggregate, in output order.

    Yields:
        Metric names, score totals `[cases, models, metrics]`, and prefix counts `[cases]`.

    Raises:
        ValueError: If required columns are missing, prefix keys are invalid or differ between
            models, or scores are nonfinite.
    """
    values, prefixes = _aligned(dataset=dataset, files=files, metrics=metrics)
    if not np.isfinite(values).all():
        raise ValueError(f'{dataset} has nonfinite scores for {", ".join(metrics)}.')
    cases = prefixes.get_level_values('case_id').to_numpy()
    starts = np.flatnonzero(np.r_[True, cases[1:] != cases[:-1]])
    totals = np.add.reduceat(values, starts, axis=0)
    counts = np.diff(np.r_[starts, len(cases)])
    yield list(metrics), totals, counts
