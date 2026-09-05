from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import Axis, read_prefix_scores, require_columns
from src.uncertainty.resampling import Units

# Which length column each breakdown groups the prefixes by. `Axis.OVERALL` is not one of them, for
# the reason `src.uncertainty.intervals.read_intervals` refuses it.
LENGTHS = {Axis.PREFIX: 'prefix_len', Axis.SUFFIX: 'suffix_len'}


def by_length(
    frame: pd.DataFrame, metrics: Sequence[str], *, axis: Axis
) -> Iterator[tuple[int, np.ndarray, Units]]:
    """Cut one run's prefixes into the units a band at each length is drawn from.

    Within one length a case has exactly one prefix, so the rows of a bucket are independent and
    each is its own unit.

    Args:
        frame: That run's per-prefix scores, from `read_prefix_scores`, carrying `metrics` and the
            length column this breakdown groups by.
        metrics: Which metrics to carry, in the order the units' trailing axis holds them.
        axis: Which breakdown to group by, one of the two `LENGTHS` names.
    Yields:
        Each length in ascending order, the metrics that survived the finite check there, and that
        bucket's prefixes as units. A length whose every metric went missing is skipped rather than
        yielded empty.
    """
    for length, part in frame.groupby(LENGTHS[axis], sort=True):
        values = part[list(metrics)].to_numpy(dtype=np.float64)
        # A metric a run scored nowhere in this bucket has no mean to bound, and a NaN anywhere in
        # the column would poison every resample of it. Checked column by column, since the metrics
        # of one bucket rarely go missing together.
        finite = np.isfinite(values).all(axis=0)
        if not finite.any():
            continue
        yield int(length), np.asarray(metrics)[finite], Units.of_rows(values[:, finite])


def _aligned(
    dataset: str, files: dict[str, Path], metrics: Sequence[str]
) -> tuple[np.ndarray, pd.MultiIndex]:
    """Read every model of one log onto the prefixes all of them scored.

    Args:
        dataset: The log these runs belong to, for the errors below.
        files: The per-prefix scores of each of its models.
        metrics: Which metrics to read, in the order the last axis holds them.
    Returns:
        Every model's score for every shared prefix, `[prefixes, models, metrics]` with the models
        in the order `files` gives them, and which prefixes those rows are, sorted so that a case's
        cuts are contiguous.
    Raises:
        ValueError: If a file scores one prefix twice, if it is missing a metric, or if the models
            of the log do not score exactly the same prefixes.
    """
    frames: dict[str, pd.DataFrame] = {}
    for model, file in files.items():
        require_columns(file, metrics)
        frame = read_prefix_scores(file).set_index(['case_id', 'prefix_len'])
        if frame.index.has_duplicates:
            raise ValueError(f'{file} scores the same prefix twice, so it cannot be compared.')
        frames[model] = frame

    shared = None
    for frame in frames.values():
        shared = frame.index if shared is None else shared.intersection(frame.index)
    assert shared is not None

    # Every model has to have answered exactly the same prefixes, not merely to overlap. A cell is
    # a mean over the prefixes its own run scored, so models scored on different sets of them are
    # already not comparable by the numbers, whatever the emphasis says.
    differing = {model: len(frame) for model, frame in frames.items() if len(frame) != len(shared)}
    if differing:
        counted = ', '.join(f'{model} {count:,}' for model, count in differing.items())
        raise ValueError(
            f'the runs of {dataset} do not score the same prefixes: {counted}, against '
            f'{len(shared):,} in common. A table compares means over the same prefixes, so '
            'generate and score every model of a log from the same test split.'
        )

    # Sorted by case first, so a case's cuts sit together and the reduction below can sum them with
    # a single pass rather than a grouping.
    shared = shared.sort_values()
    # [prefixes, models, metrics]
    values = np.stack(
        [frames[model].loc[shared, list(metrics)].to_numpy(dtype=np.float64) for model in files],
        axis=1,
    )
    return values, shared


def by_case(dataset: str, files: dict[str, Path], metrics: Sequence[str]) -> Units:
    """Reduce one log's prefixes to the cases they were cut from, which is what a test draws.

    Over the whole split a case contributes one prefix per cut point, so its prefixes are not
    independent of each other and the unit drawn has to be the whole case.

    Args:
        dataset: The log these runs belong to.
        files: The per-prefix scores of each of its models.
        metrics: Which metrics to carry, in the order the units' last axis holds them.
    Returns:
        Each case's summed scores, `[cases, models, metrics]`, weighted by how many cuts it has.
        Summed rather than averaged: every prefix weighs the same in the mean a report holds, so a
        long case weighs more, and the test has to read the same way.
    Raises:
        ValueError: If a file scores one prefix twice, if it is missing a metric, or if the models
            of the log do not score exactly the same prefixes.
    """
    values, prefixes = _aligned(dataset=dataset, files=files, metrics=metrics)
    cases = prefixes.get_level_values('case_id').to_numpy()
    starts = np.flatnonzero(np.r_[True, cases[1:] != cases[:-1]])
    return Units.of_clusters(
        totals=np.add.reduceat(values, starts, axis=0),
        weights=np.diff(np.r_[starts, len(cases)]),
    )
