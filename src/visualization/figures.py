import math
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from src.evaluation.report import Axis
from src.visualization import labels
from src.visualization.catalogue import MetricEntry, Plot
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


def _draw_metric(axes: Axes, frame: pd.DataFrame, entry: MetricEntry, *, x_bins: int | str) -> int:
    """Draw one metric onto one set of axes, a line per model, over one log's rows.

    Args:
        axes: The panel to draw onto.
        frame: The rows of one log and one breakdown, from `read_reports`.
        entry: The metric to draw, and what this figure calls it.
        x_bins: How many ticks the x-axis is allowed.
    Returns:
        The longest length any model reports here, so the panels drawn over the same lengths can
        end where the longest of them does.
    """
    # Retrieve the rows of one metric
    values = frame[frame['metric'] == entry.key]
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
    # Either end the metric leaves open is left to the data, matplotlib scaling it as it would.
    bottom, top = entry.bounds
    axes.set_ylim(bottom=bottom, top=top)
    return longest


def _link_x_axes(grid: np.ndarray, breakdowns: list[Axis], longest: list[list[int]]) -> None:
    """Put the panels of one column drawn against one breakdown on one x-axis.

    By column and breakdown, not by row: a column is one log, and two logs are two processes whose
    lengths are not on one scale, while a prefix length and a suffix length are two different
    quantities of the one log.

    Args:
        grid: The figure's panels, a row per metric and breakdown and a column per log.
        breakdowns: The breakdown each row of the grid is drawn against.
        longest: The longest length drawn in each panel, from `_draw_metric`.
    """
    for column in range(grid.shape[1]):
        for breakdown in dict.fromkeys(breakdowns):
            rows = [row for row, drawn in enumerate(breakdowns) if drawn == breakdown]
            # The axis ends where the data does rather than at the padding matplotlib would leave
            # past it, and every panel over these lengths ends there together.
            right = max(longest[row][column] for row in rows)
            for row in rows:
                grid[row][column].set_xlim(left=1, right=right)
            # The lowest of them prints the lengths for all of them, the rest being the same ticks
            # in the height of a panel.
            for row in rows[:-1]:
                grid[row][column].tick_params(labelbottom=False)


def compose_figure(frame: pd.DataFrame, plot: Plot) -> Figure:
    """Compose one figure of the catalogue, covering every log at once, untitled since a paper
    captions its figures. A column per log and a row per metric and length breakdown.

    Args:
        frame: Every report's rows the figure is drawn from, from `read_reports`.
        plot: Which figure to draw.
    Returns:
        The finished figure.
    """
    datasets = labels.DATASETS.ordered(frame['dataset'])
    # By breakdown first, so the rows drawn over one set of lengths are a block: they share their
    # x-axis, print its ticks once at the foot of the block and are named by it there.
    rows = [(breakdown, entry) for breakdown in plot.breakdowns for entry in plot.metrics]
    # Page width once there are enough logs to fill it, and a column of panels of the usual width
    # below that, so drawing one or two logs across gives a figure of the size the same panels have
    # everywhere else rather than one panel blown up to the width of the page.
    width = min(PAGE_WIDTH, len(datasets) * COLUMN_WIDTH)
    figure, grid = plt.subplots(
        nrows=len(rows),
        ncols=len(datasets),
        figsize=(width, len(rows) * width / len(datasets) * ASPECT + LEGEND_HEIGHT),
        squeeze=False,
        constrained_layout=True,
    )
    # What each panel drew, since the x-axis a block of rows ends at is the longest of them.
    longest = []
    for (breakdown, entry), row in zip(rows, grid, strict=True):
        drawn = frame[frame['axis'] == breakdown]
        longest.append(
            [
                _draw_metric(axes, drawn[drawn['dataset'] == dataset], entry, x_bins=PANEL_X_BINS)
                for axes, dataset in zip(row, datasets, strict=True)
            ]
        )
        # The metric names the row it is drawn along, its unit included, the way it names the
        # y-axis of a single-panel figure.
        # Wrapped to the width a panel title is: a row is only as tall as one panel, and a metric
        # whose name and unit run past that would otherwise be set taller than what it labels.
        row[0].set_ylabel(textwrap.fill(entry.axis_label, width=TITLE_WIDTH))
        if entry.shares_scale:
            # A row drawn over one fixed range in every panel has ticks that are the metric's and
            # not each log's: printing them once says the same thing in the width of one panel
            # less.
            for axes in row[1:]:
                axes.tick_params(labelleft=False)

    _link_x_axes(grid, [breakdown for breakdown, _ in rows], longest)

    # The logs title the top row alone: the column below a title is one log throughout.
    for axes, dataset in zip(grid[0], datasets, strict=True):
        axes.set_title(labels.DATASETS[dataset])

    if len(plot.breakdowns) == 1:
        figure.supxlabel(AXIS_LABELS[plot.breakdowns[0]])
    else:
        # One label under the whole figure could not name two breakdowns, so each block is named
        # at its own foot, under the panels whose lengths it is naming.
        for breakdown in plot.breakdowns:
            foot = max(row for row, (drawn, _) in enumerate(rows) if drawn == breakdown)
            for axes in grid[foot]:
                axes.set_xlabel(AXIS_LABELS[breakdown])

    if len(labels.MODELS.ordered(frame['model'])) > 1:
        legend_above(figure, *grid[0][0].get_legend_handles_labels())
    return figure
