from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.evaluation.prefix_scores import read_prefix_scores, score_files
from src.evaluation.report import Axis

# The probabilities the quantiles of a spread are taken at, evenly spaced so a band between any two
# of them is read the same way whichever pair a figure picks. Includes the quartiles exactly, which
# is what a box and an inter-quartile band are drawn from.
PROBABILITIES = tuple(round(0.025 * step, 3) for step in range(41))
_Q1, _MEDIAN, _Q3 = (PROBABILITIES.index(p) for p in (0.25, 0.5, 0.75))

# How many points the density of a spread is evaluated at, and how far past the box a whisker
# reaches before a value is left out of it, in inter-quartile ranges. The whisker is Tukey's own,
# which is what a reader takes a box to mean.
GRID = 128
WHISKER_REACH = 1.5
# The bandwidth the density is smoothed at, as Scott's rule scales it. A histogram on the grid
# smoothed by a Gaussian of this width, rather than a kernel evaluated against every value: a run
# answers a quarter of a million prefixes, and the two agree to well inside a line's width.
SCOTT_FACTOR = 1.06

# Which length column each breakdown groups the prefixes by. `Axis.OVERALL` groups by nothing,
# every prefix of the run being one spread.
_LENGTHS = {Axis.PREFIX: 'prefix_len', Axis.SUFFIX: 'suffix_len'}

# What `read_spreads` returns. The key columns of `read_reports` exactly, so a figure filters the
# two the same way and a spread can be drawn beside the mean it belongs to. How many prefixes the
# row covers is carried by the `Spread` itself rather than by a column of its own.
SPREAD_COLUMNS = ('dataset', 'model', 'axis', 'length', 'metric', 'spread')


@dataclass(frozen=True, slots=True)
class Spread:
    """How one run's scores for one metric are distributed over the prefixes it answered.

    Everything a page can draw of a distribution, computed once here: the quantiles a box and an
    inter-quartile band are read off, the whiskers, and a density a violin is drawn from. A figure
    does no statistics of its own, so a box, a violin and a band around a line are three ways of
    drawing one thing rather than three computations that could drift apart.
    """

    count: int
    mean: float
    # At `PROBABILITIES`, so `quantiles[0]` is the minimum and `quantiles[-1]` the maximum.
    quantiles: tuple[float, ...]
    # The furthest value still within `WHISKER_REACH` inter-quartile ranges of its quartile, which
    # is a value the run actually scored rather than the quartile pushed out by a fixed amount.
    whisker_low: float
    whisker_high: float
    # The density evaluated over `grid`, at its own peak of 1.0: a violin is drawn to the width of
    # its panel's slot whatever the metric measures, so what a figure needs is the shape and not
    # the scale. A metric whose values are all one number has a single point in both.
    grid: tuple[float, ...]
    density: tuple[float, ...]

    @property
    def minimum(self) -> float:
        """The lowest value the run scored."""
        return self.quantiles[0]

    @property
    def maximum(self) -> float:
        """The highest value the run scored."""
        return self.quantiles[-1]

    @property
    def q1(self) -> float:
        """The lower quartile, the foot of a box and of an inter-quartile band."""
        return self.quantiles[_Q1]

    @property
    def median(self) -> float:
        """The median, the line drawn across a box."""
        return self.quantiles[_MEDIAN]

    @property
    def q3(self) -> float:
        """The upper quartile, the top of a box and of an inter-quartile band."""
        return self.quantiles[_Q3]


def _density(values: np.ndarray, quantiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The shape of a distribution over a fixed grid, at its own peak of 1.0.

    Args:
        values: One run's score for the metric on every prefix it answered.
        quantiles: Those values' quantiles at `PROBABILITIES`, whose ends are the range the grid
            spans.
    Returns:
        The grid and the density over it. A metric whose values are all one number has a single
        point in each, there being no range to spread a density over.
    """
    low, high = float(quantiles[0]), float(quantiles[-1])
    if not high > low:
        return np.array([low]), np.array([1.0])

    edges = np.linspace(low, high, GRID + 1)
    counts, _ = np.histogram(values, bins=edges)
    grid = 0.5 * (edges[:-1] + edges[1:])

    # Scott's rule, in bins rather than in the metric's own units, so the smoothing is the same
    # shape whatever the metric measures.
    width = (high - low) / GRID
    bandwidth = SCOTT_FACTOR * float(values.std()) * values.size ** (-0.2) / width
    if bandwidth >= 0.5:
        # A kernel narrower than half a bin would leave the histogram as it is, spikes included,
        # which is what a near-discrete metric should look like.
        reach = int(np.ceil(3.0 * bandwidth))
        offsets = np.arange(-reach, reach + 1)
        kernel = np.exp(-0.5 * (offsets / bandwidth) ** 2)
        counts = np.convolve(counts, kernel / kernel.sum(), mode='same')

    peak = counts.max()
    return grid, counts / peak if peak > 0 else counts


def _spread(values: np.ndarray) -> Spread:
    """Reduce one run's scores for one metric to everything a figure can draw of them.

    Args:
        values: The finite scores of every prefix the row covers.
    Returns:
        The quantiles, the whiskers, the mean and the density.
    """
    quantiles = np.percentile(values, [100.0 * p for p in PROBABILITIES])
    q1, q3 = float(quantiles[_Q1]), float(quantiles[_Q3])
    reach = WHISKER_REACH * (q3 - q1)
    below = values[values >= q1 - reach]
    above = values[values <= q3 + reach]
    grid, density = _density(values, quantiles)
    return Spread(
        count=int(values.size),
        mean=float(values.mean()),
        quantiles=tuple(float(value) for value in quantiles),
        whisker_low=float(below.min()) if below.size else q1,
        whisker_high=float(above.max()) if above.size else q3,
        grid=tuple(float(value) for value in grid),
        density=tuple(float(value) for value in density),
    )


def _rows(
    frame: pd.DataFrame, metrics: Sequence[str], axes: Sequence[Axis], identity: dict[str, str]
) -> list[dict[str, object]]:
    """Lay one run's per-prefix scores out as one row per metric per breakdown.

    Args:
        frame: That run's per-prefix scores, from `read_prefix_scores`.
        metrics: Which metrics to reduce.
        axes: Which breakdowns to reduce them over.
        identity: The dataset and model every row of this run carries.
    Returns:
        One row per metric per breakdown, under `SPREAD_COLUMNS`.
    """
    rows: list[dict[str, object]] = []
    for axis in axes:
        # One group per length, or the whole run as a single group with no length to name.
        if axis is Axis.OVERALL:
            groups: list[tuple[int | None, pd.DataFrame]] = [(None, frame)]
        else:
            groups = [(int(length), part) for length, part in frame.groupby(_LENGTHS[axis])]

        for length, part in groups:
            for metric in metrics:
                values = part[metric].to_numpy(dtype=np.float64)
                values = values[np.isfinite(values)]
                if not values.size:
                    continue
                rows.append(
                    {
                        **identity,
                        'axis': axis,
                        'length': length,
                        'metric': metric,
                        'spread': _spread(values),
                    }
                )
    return rows


def read_spreads(
    reports: Sequence[Path],
    *,
    metrics: Sequence[str],
    axes: Sequence[Axis] = tuple(Axis),
) -> pd.DataFrame:
    """Read how widely each run's scores are spread, from the per-prefix scores beside its report.

    A report holds the mean of every metric and says nothing about how much of the split sits with
    it. That is what the per-prefix scores already hold, and this is where they are reduced to it,
    beside the module that owns their file rather than beside whichever page draws them: a spread
    is a property of a run's scores, so a box, a violin and a band around a line all read one
    computation.

    Args:
        reports: The evaluation reports being read. Each one's per-prefix scores are read from
            beside it, which is where `pipelines.evaluate` writes them.
        metrics: Which metrics of `src.evaluation.scores.METRICS` to reduce. Named by the caller
            rather than taken to be all of them, a page usually drawing a handful.
        axes: Which breakdowns to reduce them over, the same three `read_reports` is keyed by.
            Every one costs another pass, so a caller after the overall spread alone asks for it.
    Returns:
        One row per metric per breakdown per run, under `SPREAD_COLUMNS`. `length` is a nullable
        integer, null on the `Axis.OVERALL` rows, exactly as in `read_reports`.
    Raises:
        ValueError: If a report has no per-prefix scores beside it, if one is not a scores file, if
            one log is given two runs of the same model, or if a run predates a metric asked for.
    """
    wanted = tuple(dict.fromkeys(metrics))
    needed = (*wanted, *(_LENGTHS[axis] for axis in axes if axis is not Axis.OVERALL))

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

    spreads = pd.DataFrame(rows, columns=list(SPREAD_COLUMNS))
    spreads['length'] = spreads['length'].astype('Int64')
    return spreads
