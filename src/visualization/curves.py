import math
import textwrap
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from src.evaluation.metrics import ByLengthMetrics, EvaluationMetrics
from src.visualization.metrics import MetricSpec, axis_label, direction_marker, metric_value
from src.visualization.runs import PlottedRun
from src.visualization.style import COLUMN_WIDTH, PAGE_WIDTH, figure_size

# How many panels a grid figure puts side by side, and how wide a panel title runs before it wraps
# onto a second line. Four across a two-column page fits both metric families in two rows, which is
# what keeps the figure short enough to sit on a page beside its caption.
GRID_COLUMNS = 4
GRID_TITLE_WIDTH = 18
GRID_X_BINS = 5
# How tall one row of panels is, in inches, with room for a title of up to two lines.
GRID_ROW_HEIGHT = 1.7

# The most markers to draw on one line. Beyond this they merge into the line and stop saying which
# series they belong to.
MAX_MARKERS = 12


@dataclass(frozen=True)
class LengthAxis:
    """Which per-length breakdown a figure reads, and what its x-axis is called.

    Both breakdowns an evaluation reports share this shape, so one set of figures can draw either
    by reading through this rather than a metric's length directly.
    """

    label: str  # The x-axis label of a figure drawn against this breakdown
    breakdown: Callable[[EvaluationMetrics], list[ByLengthMetrics]]


BY_PREFIX_LENGTH = LengthAxis(
    label='Prefix length', breakdown=lambda metrics: metrics.by_prefix_length
)
BY_SUFFIX_LENGTH = LengthAxis(
    label='Suffix length', breakdown=lambda metrics: metrics.by_suffix_length
)


def coverage_cutoff(runs: Sequence[PlottedRun], coverage: float, axis: LengthAxis) -> int:
    """The longest length worth drawing, given how few pairs the tail holds.

    Lengths thin out sharply: the last few of a split are averages over a handful of pairs, and
    drawing them squeezes the lengths that carry the split into the left edge of the figure.

    Args:
        runs: The runs sharing the figure, whose x-axis this bounds.
        coverage: The share of a run's pairs the axis must reach. 1.0 keeps every length.
        axis: Which breakdown the figure is drawn against.
    Returns:
        The length at which the slowest run of the set has covered `coverage` of its pairs, so no
        run is cut shorter than it needs to be.
    """
    return max(_run_cutoff(run, coverage, axis) for run in runs)


def _run_cutoff(run: PlottedRun, coverage: float, axis: LengthAxis) -> int:
    """The shortest length at which one run has covered `coverage` of its pairs."""
    breakdown = axis.breakdown(run.report.metrics)
    total = sum(length.pairs_count for length in breakdown)
    seen = 0
    for length in breakdown:
        seen += length.pairs_count
        if seen >= coverage * total:
            return length.length
    return breakdown[-1].length


def _series(
    run: PlottedRun, spec: MetricSpec, cutoff: int, axis: LengthAxis
) -> tuple[list[int], list[float]]:
    """One run's values of a metric against a length breakdown, up to the cutoff."""
    breakdown = [length for length in axis.breakdown(run.report.metrics) if length.length <= cutoff]
    return (
        [length.length for length in breakdown],
        [metric_value(length.scores, spec) for length in breakdown],
    )


def _draw(
    axes: Axes,
    runs: Sequence[PlottedRun],
    spec: MetricSpec,
    cutoff: int,
    axis: LengthAxis,
    *,
    x_bins: int | str = 'auto',
) -> None:
    """Draw every run's curve for one metric onto one set of axes.

    Args:
        axes: Where to draw.
        runs: The runs to compare, all of them scored on the same dataset.
        spec: The metric to plot.
        cutoff: The longest length to draw.
        axis: Which breakdown the curve is drawn against.
        x_bins: How many intervals to divide the x-axis into. A panel of a grid is half the width
            of a figure of its own and fits half the ticks.
    """
    for run in runs:
        lengths, values = _series(run, spec, cutoff, axis)
        axes.plot(
            lengths,
            values,
            label=run.label,
            color=run.style.color,
            marker=run.style.marker,
            linestyle=run.style.linestyle,
            markevery=max(1, math.ceil(len(lengths) / MAX_MARKERS)),
        )
    axes.xaxis.set_major_locator(MaxNLocator(nbins=x_bins, integer=True))
    axes.set_xlim(left=1, right=cutoff)
    if spec.unit is None:
        axes.set_ylim(0, 1)
    else:
        axes.set_ylim(bottom=0)


def length_curve(
    runs: Sequence[PlottedRun], spec: MetricSpec, cutoff: int, axis: LengthAxis
) -> Figure:
    """A single-column figure of one metric against a length breakdown, one line per run.

    Args:
        runs: The runs to compare, all of them scored on the same dataset.
        spec: The metric to plot.
        cutoff: The longest length to draw, from `coverage_cutoff`.
        axis: Which breakdown to draw the metric against.
    Returns:
        The finished figure, untitled since a paper captions its figures.
    """
    figure, axes = plt.subplots(figsize=figure_size(COLUMN_WIDTH), constrained_layout=True)
    _draw(axes, runs, spec, cutoff, axis)
    axes.set_xlabel(axis.label)
    axes.set_ylabel(axis_label(spec))
    # One line needs no legend: the caption names it.
    if len(runs) > 1:
        axes.legend(loc='best')
    return figure


def metric_grid(
    runs: Sequence[PlottedRun], specs: Sequence[MetricSpec], cutoff: int, axis: LengthAxis
) -> Figure:
    """A page-wide panel of several metrics against a length breakdown, sharing one legend.

    Args:
        runs: The runs to compare, all of them scored on the same dataset.
        specs: The metrics to panel, one each, laid out in the order given.
        cutoff: The longest length to draw, from `coverage_cutoff`.
        axis: Which breakdown to draw every panel against.
    Returns:
        The finished figure.
    """
    rows = math.ceil(len(specs) / GRID_COLUMNS)
    figure, grid = plt.subplots(
        nrows=rows,
        ncols=GRID_COLUMNS,
        figsize=(PAGE_WIDTH, GRID_ROW_HEIGHT * rows),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    panels = grid.flatten()

    for index, spec in enumerate(specs):
        axes = panels[index]
        _draw(axes, runs, spec, cutoff, axis, x_bins=GRID_X_BINS)
        title = f'{spec.label}{direction_marker(spec)}'
        axes.set_title(textwrap.fill(title, width=GRID_TITLE_WIDTH))
        if spec.unit is not None:
            axes.set_ylabel(spec.unit)
        # A shared x-axis labels the bottom row alone, which leaves the last panel of a short
        # column without ticks. Give them back, since that panel's axis ends there.
        if index + GRID_COLUMNS >= len(specs):
            axes.tick_params(labelbottom=True)

    for empty in panels[len(specs) :]:
        figure.delaxes(empty)

    figure.supxlabel(axis.label)
    # Above the panels, since the bottom of the figure is where the shared x-axis label sits.
    if len(runs) > 1:
        handles, labels = panels[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc='outside upper center', ncols=len(runs))
    return figure


def support_curve(
    runs: Sequence[PlottedRun], cutoff: int, coverage: float, axis: LengthAxis
) -> Figure:
    """How many pairs each length was averaged over, and where the other figures stop.

    Every length is drawn, cutoff included, since this is the figure the cutoff is read off.

    Args:
        runs: The runs to compare, all of them scored on the same dataset.
        cutoff: The cutoff the other figures of this dataset apply.
        coverage: The share of pairs that cutoff was chosen to reach, named in its legend entry.
        axis: Which breakdown the support is drawn against.
    Returns:
        The finished figure.
    """
    figure, axes = plt.subplots(figsize=figure_size(COLUMN_WIDTH), constrained_layout=True)
    for run in runs:
        breakdown = axis.breakdown(run.report.metrics)
        lengths = [length.length for length in breakdown]
        axes.plot(
            lengths,
            [length.pairs_count for length in breakdown],
            label=run.label,
            color=run.style.color,
            marker=run.style.marker,
            linestyle=run.style.linestyle,
            markevery=max(1, math.ceil(len(lengths) / MAX_MARKERS)),
        )
    axes.axvline(
        x=cutoff,
        color='#444444',
        linewidth=0.8,
        linestyle=(0, (2, 2)),
        label=f'{coverage:.0%} of pairs',
    )
    axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes.set_xlim(left=1)
    axes.set_ylim(bottom=0)
    axes.set_xlabel(axis.label)
    axes.set_ylabel('Pairs scored')
    axes.legend(loc='best')
    return figure
