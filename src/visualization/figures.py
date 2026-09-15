import math
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from src.evaluation import Axis
from src.scalar_metrics import Owner
from src.visualization import labels
from src.visualization.catalogue import MetricEntry, Plot
from src.visualization.style import (
    DATASET_HEIGHT,
    FIGURE_OVERHEAD,
    MAX_MARKERS,
    PAGE_WIDTH,
    PANEL_X_BINS,
    TITLE_WIDTH,
    Y_HEADROOM,
    legend_above,
)

# X-axis labels by length breakdown.
AXIS_LABELS = {Axis.PREFIX: 'Prefix length', Axis.SUFFIX: 'Suffix length'}


def _draw_metric(axes: Axes, frame: pd.DataFrame, entry: MetricEntry) -> int:
    """Draw one metric.

    Args:
        axes: Panel to draw on.
        frame: Report rows for one dataset and breakdown.
        entry: Metric display definition.

    Returns:
        Longest reported length.
    """
    # Select rows for the requested metric.
    values = frame[frame['metric'] == entry.key]
    drawn = labels.MODELS.ordered(values['model'])
    # Log-owned metrics produce one shared series.
    if entry.metric.owner is Owner.LOG:
        series = [(model, labels.LOG_STYLE) for model in drawn[:1]]
    else:
        series = [(model, labels.MODELS[model]) for model in drawn]

    lines = [
        (values[values['model'] == model].sort_values('length'), style) for model, style in series
    ]

    longest = 1
    for line, style in lines:
        longest = max(longest, int(line['length'].max()))
        axes.plot(
            line['length'],
            line['value'],
            label=style.label,
            color=style.color,
            marker=style.marker,
            linestyle=style.linestyle,
            markevery=max(1, math.ceil(len(line) / MAX_MARKERS)),
        )
    return longest


def _draw_panel(
    axes: Axes, frame: pd.DataFrame, panel: tuple[MetricEntry, ...], *, x_bins: int | str
) -> int:
    """Draw a panel and return its longest reported length.

    Args:
        axes: Panel to draw on.
        frame: Report rows for one dataset and breakdown.
        panel: Metrics displayed together.
        x_bins: Maximum x-axis tick bins.

    Returns:
        Longest reported length across the panel.
    """
    longest = max(_draw_metric(axes, frame, entry) for entry in panel)
    axes.xaxis.set_major_locator(MaxNLocator(nbins=x_bins, integer=True))
    # Unbounded sides use Matplotlib's automatic limits.
    bottom, top = panel[0].bounds
    if top is not None:
        # Keep values at the bound from appearing clipped.
        top += Y_HEADROOM * (top - (bottom if bottom is not None else 0.0))
    axes.set_ylim(bottom=bottom, top=top)
    if panel[0].shares_scale:
        axes.set_yticks([0.0, 0.5, 1.0])
    return longest


def _link_x_axes(grid: np.ndarray, breakdowns: list[Axis], longest: list[list[int]]) -> None:
    """Share x-axis limits within each dataset and breakdown.

    Args:
        grid: Figure axes indexed by dataset and metric column.
        breakdowns: Breakdown represented by each column.
        longest: Longest reported length for each panel.
    """
    for row in range(grid.shape[0]):
        for breakdown in dict.fromkeys(breakdowns):
            columns = [column for column, drawn in enumerate(breakdowns) if drawn == breakdown]
            # Align panel limits to the longest observed series.
            right = max(longest[row][column] for column in columns)
            for column in columns:
                grid[row][column].set_xlim(left=1, right=max(2, right))


def compose_figure(frame: pd.DataFrame, plot: Plot) -> Figure:
    """Compose a catalogue figure across all datasets.

    Args:
        frame: Report rows.
        plot: Figure definition.

    Returns:
        Composed Matplotlib figure.
    """
    datasets = labels.DATASETS.ordered(frame['dataset'])
    # Each breakdown occupies a block of metric columns.
    columns = [(breakdown, panel) for breakdown in plot.breakdowns for panel in plot.panels]
    figure, grid = plt.subplots(
        nrows=len(datasets),
        ncols=len(columns),
        figsize=(PAGE_WIDTH, len(datasets) * DATASET_HEIGHT + FIGURE_OVERHEAD),
        squeeze=False,
        constrained_layout=True,
    )
    # Track panel extents for linked x-axes.
    longest = []
    for dataset, row in zip(datasets, grid, strict=True):
        drawn = frame[frame['dataset'] == dataset]
        longest.append(
            [
                _draw_panel(axes, drawn[drawn['axis'] == breakdown], panel, x_bins=PANEL_X_BINS)
                for axes, (breakdown, panel) in zip(row, columns, strict=True)
            ]
        )
        row[0].set_ylabel(
            labels.DATASETS[dataset], rotation=0, ha='right', va='center', labelpad=12
        )

    figure.align_ylabels(grid[:, 0])
    _link_x_axes(grid, [breakdown for breakdown, _ in columns], longest)

    for column, (_, panel) in enumerate(columns):
        heading = panel[0].axis_label.replace(' (', '\n(')
        grid[0, column].set_title(
            '\n'.join(textwrap.fill(line, width=TITLE_WIDTH) for line in heading.splitlines())
        )
        # A metric has the same scale across datasets, including unbounded metrics.
        limits = [axes.get_ylim() for axes in grid[:, column]]
        bottom = min(limit[0] for limit in limits)
        top = max(limit[1] for limit in limits)
        for axes in grid[:, column]:
            axes.set_ylim(bottom, top)
            if column > 0 and panel[0].shares_scale and columns[0][1][0].shares_scale:
                axes.tick_params(labelleft=False)

    if len(plot.breakdowns) == 1:
        figure.supxlabel(AXIS_LABELS[plot.breakdowns[0]])
    else:
        for axes, (breakdown, _) in zip(grid[-1], columns, strict=True):
            axes.set_xlabel(AXIS_LABELS[breakdown])

    # Deduplicate legend entries across panels.
    keys: dict[str, Artist] = {}
    for row in grid:
        for axes in row:
            handles, written = axes.get_legend_handles_labels()
            for label, handle in zip(written, handles, strict=True):
                keys.setdefault(label, handle)
    if len(keys) > 1:
        legend_above(figure, list(keys.values()), list(keys))
    return figure
