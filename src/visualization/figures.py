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
    figure_size,
    legend_above,
)

# What the x-axis of a figure drawn against each length breakdown is called
AXIS_LABELS = {Axis.PREFIX: 'Prefix length', Axis.SUFFIX: 'Suffix length'}


@dataclass(frozen=True)
class Plot:
    """One figure of the catalogue. The group is what the figure is about,
    the axis is which breakdown it shows, and the metrics are the panels it draws."""

    group: str  # What the figure is about, and the first half of the file it is written to
    axis: Axis  # Which breakdown, `Axis.PREFIX` or `Axis.SUFFIX`
    metrics: tuple[str, ...]  # One panel each, in the order they are laid out

    @property
    def name(self) -> str:
        """What the figure is written as, e.g. `dls-by-prefix-length`."""
        return f'{self.group}-by-{self.axis}-length'


# Every figure of the catalogue, each a group of metrics against one length breakdown, drawn one
# panel per metric and one line per model. A new figure is an entry here and nothing else.
#
# A figure is worth a page only where it shows what a table cell cannot: a shape over length. Which
# breakdown carries that shape is the group's own question, so most groups are drawn against one of
# the two rather than both. The two axes are not independent: every prefix of a case is scored, so a
# long prefix leaves a short suffix, and a curve rising with prefix length is partly a curve over
# how much is left to predict. `Axis.SUFFIX` is what a group is drawn against where it is about how
# far a model can generate, and `Axis.PREFIX` where it is about how much a model is told first.
FIGURES = (
    # Both breakdowns, alone among the groups: the accuracy of a suffix is what the paper is about,
    # and the two axes are the two readings of it. The three panels are the point estimate, the
    # mean of the samples and the closest of them, so the gap between the last two is what drawing
    # more than one suffix buys.
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
    # Against the suffix: whether a model stays inside the process as it generates further is a
    # question about how much it has generated, not about how much it was given.
    Plot(
        group='conformance',
        axis=Axis.SUFFIX,
        metrics=('conformance_point', 'conformance_mean'),
    ),
    # Against the prefix: how early in a case its remaining time can be trusted. Against the suffix
    # it would mostly redraw its own axis, an error in days growing with the days left to predict.
    Plot(
        group='remaining-time',
        axis=Axis.PREFIX,
        metrics=('remaining_time_ae_point_days', 'remaining_time_ae_mean_days'),
    ),
    # Against the prefix: how much of a case has to be seen before the spread of `p(z | prefix)`
    # closes in.
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
    """Compose one figure of the catalogue, with one panel per metric and one line per model.

    Args:
        frame: The rows of one log, from `read_reports`.
        plot: Which figure to draw.
    Returns:
        The finished figure, untitled since a paper captions its figures: one panel per metric at
        page width, or a single column figure where the group holds one metric alone.
    """
    # Compute the number of panels first
    num_panels = len(plot.metrics)

    # A row of panels is as tall as one panel drawn at the shared aspect, plus the room the legend
    # and the titles take above them.
    size = (
        figure_size(COLUMN_WIDTH)
        if num_panels == 1
        else (PAGE_WIDTH, PAGE_WIDTH / num_panels * ASPECT + LEGEND_HEIGHT)
    )
    figure, grid = plt.subplots(
        nrows=1,
        ncols=num_panels,
        figsize=size,
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )

    for axes, metric_key in zip(grid[0], plot.metrics, strict=True):
        _draw_metric(axes, frame, metric_key, x_bins='auto' if num_panels == 1 else PANEL_X_BINS)
        metric = labels.METRICS[metric_key]
        if num_panels == 1:
            axes.set_ylabel(metric.axis_label)
        else:
            axes.set_title(textwrap.fill(metric.title, width=TITLE_WIDTH))
            # The unit alone, the panel's title already naming what it measures.
            if metric.unit is not None:
                axes.set_ylabel(metric.unit)

    models = labels.MODELS.ordered(frame['model'])
    if num_panels == 1:
        grid[0][0].set_xlabel(AXIS_LABELS[plot.axis])
        # One line needs no legend: the caption names it.
        if len(models) > 1:
            grid[0][0].legend(loc='best')
        return figure

    figure.supxlabel(AXIS_LABELS[plot.axis])
    # Above the panels, since the bottom of the figure is where the shared x-axis label sits.
    if len(models) > 1:
        legend_above(figure, *grid[0][0].get_legend_handles_labels())
    return figure
