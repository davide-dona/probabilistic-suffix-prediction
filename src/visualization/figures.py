import math
import textwrap
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from src.evaluation.report import Axis
from src.visualization import labels
from src.visualization.style import (
    ASPECT,
    COLUMN_WIDTH,
    LEGEND_HEIGHT,
    MAX_MARKERS,
    PAGE_WIDTH,
    PANEL_X_BINS,
    TITLE_WIDTH,
    legend_above,
)

# What the x-axis of a figure drawn against each length breakdown is called
AXIS_LABELS = {Axis.PREFIX: 'Prefix length', Axis.SUFFIX: 'Suffix length'}


@dataclass(frozen=True)
class Plot:
    """One figure of the catalogue, covering every log at once. The group is what the figure is
    about, the axis is which breakdown it shows, and the metrics are what it draws."""

    group: str  # What the figure is about, and the first half of the file it is written to
    axis: Axis  # Which breakdown, `Axis.PREFIX` or `Axis.SUFFIX`
    metrics: tuple[str, ...]  # One panel each, in the order they are laid out

    @property
    def name(self) -> str:
        """What the figure is written as, e.g. `dls-by-prefix-length`."""
        return f'{self.group}-by-{self.axis}-length'


# Every figure of the catalogue, each a group of metrics against one length breakdown, drawn one
# column per log and one row per metric, a line per model within a panel. A new figure is an
# entry here and nothing else.
FIGURES = (
    Plot(
        group='dls',
        axis=Axis.PREFIX,
        metrics=('dls_point', 'dls_mean', 'dls_best'),
    ),
    Plot(
        group='dls',
        axis=Axis.SUFFIX,
        metrics=('dls_point', 'dls_mean', 'dls_best'),
    ),
    Plot(
        group='conformance',
        axis=Axis.SUFFIX,
        metrics=('conformance_point', 'conformance_mean'),
    ),
    Plot(
        group='remaining-time',
        axis=Axis.PREFIX,
        metrics=('remaining_time_ae_point_days', 'remaining_time_ae_mean_days'),
    ),
    Plot(
        group='diversity',
        axis=Axis.PREFIX,
        metrics=('sample_diversity', 'unique_sample_rate'),
    ),
)


def _draw_metric(axes: Axes, frame: pd.DataFrame, metric: str, *, x_bins: int | str) -> None:
    """Draw one metric onto one set of axes, a line per model, over one log's rows."""
    # Retrieve the rows of one metric
    values = frame[frame['metric'] == metric]
    longest = 1
    # For each model, draw its line over the lengths it reports
    for model in labels.MODELS.ordered(values['model']):
        line = values[values['model'] == model].sort_values('length')
        model_style = labels.MODELS[model]
        longest = max(longest, int(line['length'].max()))
        axes.plot(
            line['length'],
            line['value'],
            label=model_style.label,
            color=model_style.color,
            marker=model_style.marker,
            linestyle=model_style.linestyle,
            markevery=max(1, math.ceil(len(line) / MAX_MARKERS)),
        )
    axes.xaxis.set_major_locator(MaxNLocator(nbins=x_bins, integer=True))
    # The longest length any run reports, so the axis ends where the data does rather than at the
    # padding matplotlib would leave past it.
    axes.set_xlim(left=1, right=longest)
    if labels.METRICS[metric].is_score:
        axes.set_ylim(0, 1)
    else:
        axes.set_ylim(bottom=0)


def compose_figure(frame: pd.DataFrame, plot: Plot) -> Figure:
    """Compose one figure of the catalogue, covering every log at once, untitled since a paper
    captions its figures. A column per log and a row per metric.

    Args:
        frame: Every report's rows the figure is drawn from, from `read_reports`.
        plot: Which figure to draw.
    Returns:
        The finished figure.
    """
    datasets = labels.DATASETS.ordered(frame['dataset'])
    # Page width once there are enough logs to fill it, and a column of panels of the usual width
    # below that, so drawing one or two logs across gives a figure of the size the same panels have
    # everywhere else rather than one panel blown up to the width of the page.
    width = min(PAGE_WIDTH, len(datasets) * COLUMN_WIDTH)
    figure, grid = plt.subplots(
        nrows=len(plot.metrics),
        ncols=len(datasets),
        figsize=(width, len(plot.metrics) * width / len(datasets) * ASPECT + LEGEND_HEIGHT),
        # By column, since a column is one log and its panels run over that log's own lengths. Not
        # by row: two logs are two processes, and a shared scale would draw the one whose cases run
        # for weeks over the one whose cases run for hours. A metric already in [0, 1] is on the
        # whole interval in every panel either way, which is what makes those rows comparable.
        sharex='col',
        squeeze=False,
        constrained_layout=True,
    )
    for row, metric_key in zip(grid, plot.metrics, strict=True):
        for axes, dataset in zip(row, datasets, strict=True):
            _draw_metric(axes, frame[frame['dataset'] == dataset], metric_key, x_bins=PANEL_X_BINS)
        # The metric names the row it is drawn along, its unit included, the way it names the
        # y-axis of a single-panel figure.
        metric = labels.METRICS[metric_key]
        # Wrapped to the width a panel title is: a row is only as tall as one panel, and a metric
        # whose name and unit run past that would otherwise be set taller than what it labels.
        row[0].set_ylabel(textwrap.fill(metric.axis_label, width=TITLE_WIDTH))
        if metric.is_score:
            # A row of a metric already in [0, 1] is drawn over that whole interval in every panel,
            # so the ticks are the metric's and not each log's: printing them once says the same
            # thing in the width of one panel less.
            for axes in row[1:]:
                axes.tick_params(labelleft=False)

    # The logs title the top row alone: the column below a title is one log throughout.
    for axes, dataset in zip(grid[0], datasets, strict=True):
        axes.set_title(labels.DATASETS[dataset])

    figure.supxlabel(AXIS_LABELS[plot.axis])
    if len(labels.MODELS.ordered(frame['model'])) > 1:
        legend_above(figure, *grid[0][0].get_legend_handles_labels())
    return figure
