from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.evaluation.prefix_scores import read_prefix_scores, score_files
from src.evaluation.report import Axis

# What the resampling is reproducible under, so two runs of the pipeline draw the same bands and
# emphasize the same cells.
SEED = 24
# How many resamples are drawn at once. One draw is a row of `[b, units]`, so this trades the memory
# that matrix takes against the number of passes over what is being resampled.
CHUNK = 250

# What a band covers, and how many resamples its two ends are read off. Fewer than the 10,000
# `significance.py` draws, and deliberately: a p-value there has to survive being multiplied by the
# comparisons of a row, where a band is two order statistics of the resample means and is settled
# long before this. The draw itself is the cost here, `RESAMPLES` integers per prefix of every run
# and breakdown.
LEVEL = 0.95
RESAMPLES = 2_000
_TAILS = (50.0 * (1.0 - LEVEL), 50.0 * (1.0 + LEVEL))

# Which length column each breakdown groups the prefixes by. `Axis.OVERALL` is not one of them, for
# the reason `read_intervals` refuses it.
_LENGTHS = {Axis.PREFIX: 'prefix_len', Axis.SUFFIX: 'suffix_len'}

# What `read_intervals` returns. The key columns of `read_reports` exactly, minus the two a report
# already carries, so a band joins onto the mean it is drawn around on the key.
INTERVAL_COLUMNS = ('dataset', 'model', 'axis', 'length', 'metric', 'low', 'high')


def resample_counts(units: int, resamples: int, *, generator: np.random.Generator) -> Iterator:
    """Draw a bootstrap's resamples, a chunk of them at a time.

    One resample draws `units` of the units with replacement, so what it yields is how many times
    each unit was drawn rather than which ones were. A caller applies those counts to whatever the
    unit carries, which is what lets one draw serve every metric at once.

    Args:
        units: How many units there are to draw from, a prefix here and a whole case in
            `src.visualization.significance`.
        resamples: How many resamples to draw in total.
        generator: The source of the draws, seeded by the caller off `SEED`.
    Yields:
        `[chunk, units]` of counts as float64, ready to be multiplied against the units' values.
        The chunks are `CHUNK` long except the last, and sum to `resamples`.
    """
    drawn = 0
    while drawn < resamples:
        size = min(CHUNK, resamples - drawn)
        # Which unit each of `units` draws landed on, offset so that a resample's draws fall in its
        # own stretch of one flat count: `[size, units]` counted in a single `bincount`, which is an
        # order of magnitude cheaper than a multinomial over this many units.
        picks = generator.integers(low=0, high=units, size=(size, units))
        picks += (np.arange(size) * units)[:, None]
        yield (
            np.bincount(picks.ravel(), minlength=size * units)
            .reshape(size, units)
            .astype(np.float64)
        )
        drawn += size


def _interval(values: np.ndarray, *, generator: np.random.Generator) -> np.ndarray:
    """Bound how far the mean of one length's prefixes could be off, on every metric at once.

    Args:
        values: That length's scores, `[prefixes, metrics]`. One prefix is one unit: within a
            length a case has exactly one prefix, so the rows are independent and the ordinary
            bootstrap `resample_counts` draws is the right one.
        generator: The source of the draws.
    Returns:
        The two ends of the interval, `[2, metrics]`, at `_TAILS` of the resample means. Each end
        is itself a mean of the values, so a band on a metric bounded in `[0, 1]` cannot leave it.
    """
    prefixes = values.shape[0]
    # One draw serves every metric: the counts are `[chunk, prefixes]` and the values
    # `[prefixes, metrics]`, so a resample's means are one matmul rather than a pass per metric.
    means = [
        (counts @ values) / prefixes
        for counts in resample_counts(prefixes, RESAMPLES, generator=generator)
    ]
    return np.percentile(np.concatenate(means, axis=0), _TAILS, axis=0)


def _rows(
    frame: pd.DataFrame, metrics: Sequence[str], axes: Sequence[Axis], identity: dict[str, str]
) -> list[dict[str, object]]:
    """Bound one run's mean at every length of every breakdown.

    Args:
        frame: That run's per-prefix scores, from `read_prefix_scores`.
        metrics: Which metrics to bound, in the order the columns of `_interval` come back in.
        axes: Which breakdowns to bound them over.
        identity: The dataset and model every row of this run carries.
    Returns:
        One row per metric per length per breakdown, under `INTERVAL_COLUMNS`.
    """
    # Seeded per run rather than per length, so a run's bands do not depend on how many lengths
    # were asked for before them, and two runs of the pipeline still draw the same ones.
    generator = np.random.default_rng(SEED)

    rows: list[dict[str, object]] = []
    for axis in axes:
        for length, part in frame.groupby(_LENGTHS[axis], sort=True):
            values = part[list(metrics)].to_numpy(dtype=np.float64)
            # A metric a run scored nowhere in this bucket has no mean to bound, and a NaN anywhere
            # in the column would poison every resample of it. Bounded column by column, since the
            # metrics of one bucket rarely go missing together.
            finite = np.isfinite(values).all(axis=0)
            if not finite.any():
                continue
            low, high = _interval(values[:, finite], generator=generator)
            rows.extend(
                {
                    **identity,
                    'axis': axis,
                    'length': int(length),
                    'metric': metric,
                    'low': float(low[column]),
                    'high': float(high[column]),
                }
                for column, metric in enumerate(np.asarray(metrics)[finite])
            )
    return rows


def read_intervals(
    reports: Sequence[Path],
    *,
    metrics: Sequence[str],
    axes: Sequence[Axis] = (Axis.PREFIX, Axis.SUFFIX),
) -> pd.DataFrame:
    """Bound how far each length's reported mean could be off, from the per-prefix scores beside
    each report.

    A report holds the mean of every metric at every length and says nothing about how well that
    many prefixes pin it down. That is what the per-prefix scores already hold, and this is where
    they are reduced to it, beside the module that owns their file rather than beside the figure
    that draws them: how far a mean could be off is a property of the scores it was taken over, not
    a claim a page makes.

    Args:
        reports: The evaluation reports being read. Each one's per-prefix scores are read from
            beside it, which is where `pipelines.evaluate` writes them.
        metrics: Which metrics of `src.evaluation.scores.METRICS` to bound. Named by the caller
            rather than taken to be all of them, a page usually drawing a handful.
        axes: Which length breakdowns to bound them over, the two `read_reports` is keyed by that a
            figure draws a line along.
    Returns:
        One row per metric per length per breakdown per run, under `INTERVAL_COLUMNS`. `length` is
        a nullable integer, as in `read_reports`, so the two frames join on the key columns.
    Raises:
        ValueError: If `Axis.OVERALL` is asked for, if a report has no per-prefix scores beside it,
            if one is not a scores file, if one log is given two runs of the same model, or if a run
            predates a metric asked for.
    """
    if Axis.OVERALL in axes:
        raise ValueError(
            'the overall breakdown cannot be bounded by an ordinary bootstrap: over the whole '
            'split a case contributes one prefix per cut point, so the prefixes are not '
            'independent and resampling them reads the interval far too narrow. Within one length '
            'a case has exactly one prefix, which is why the two length breakdowns are bounded '
            'here; a difference over the whole split is settled by the cluster bootstrap in '
            '`src.visualization.significance` instead.'
        )

    wanted = tuple(dict.fromkeys(metrics))
    needed = (*wanted, *(_LENGTHS[axis] for axis in axes))

    rows: list[dict[str, object]] = []
    for dataset, files in score_files(reports).items():
        for model, file in files.items():
            # Checked against the schema rather than after the read, which would fail on the
            # missing column with nothing to say about why it is missing.
            absent = [key for key in needed if key not in pq.read_schema(where=file).names]
            if absent:
                raise ValueError(
                    f'{file} carries no {", ".join(absent)}, so it predates the scores now read. '
                    'Score it again with `python -m pipelines.evaluate`.'
                )
            frame = read_prefix_scores(file, columns=needed)
            rows.extend(_rows(frame, wanted, axes, {'dataset': dataset, 'model': model}))

    intervals = pd.DataFrame(rows, columns=list(INTERVAL_COLUMNS))
    return intervals.astype({'length': 'Int64', 'low': 'float64', 'high': 'float64'})
