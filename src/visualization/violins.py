import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from src.evaluation.spreads import Spread
from src.visualization import labels
from src.visualization.catalogue import Column, MetricEntry, Violin
from src.visualization.labels import ModelStyle
from src.visualization.style import (
    ASPECT,
    COLUMN_WIDTH,
    LEGEND_HEIGHT,
    PAGE_WIDTH,
    TITLE_WIDTH,
    VIOLIN_ALPHA,
    VIOLIN_WIDTH,
    Y_HEADROOM,
    legend_above,
)

# The marker the mean is drawn with inside a violin. The tables print the mean and a violin's own
# bar is the median, so without this the figure and the table would say different things about the
# same run.
MEAN_MARKER = 'x'
MEAN_SIZE = 3.2


def _draw_violin(axes: Axes, spread: Spread, *, position: float, style: ModelStyle) -> None:
    """Draw one series' violin at one slot of a panel.

    Args:
        axes: The panel to draw onto.
        spread: That series' distribution for the panel's metric, from `read_spreads`.
        position: Which slot of the panel it occupies, the same slot in every panel of the figure.
        style: The colour the series keeps throughout, a model's or the log's.
    """
    stroke = plt.rcParams['axes.linewidth']
    grid = np.asarray(spread.grid)
    # The density is already at a peak of 1.0, so a violin is drawn to the same width whatever the
    # metric measures and two of them are compared on their shape rather than on their scale.
    half = 0.5 * VIOLIN_WIDTH * np.asarray(spread.density)
    if grid.size > 1:
        axes.fill_betweenx(
            grid,
            position - half,
            position + half,
            facecolor=style.color,
            alpha=VIOLIN_ALPHA,
            edgecolor=style.color,
            linewidth=stroke,
            zorder=2,
        )

    # The quartiles and the whiskers over the body, so a violin is still read the way a box is.
    axes.vlines(
        position,
        spread.whisker_low,
        spread.whisker_high,
        color=style.color,
        linewidth=stroke,
        zorder=3,
    )
    axes.vlines(
        position,
        spread.q1,
        spread.q3,
        color=style.color,
        linewidth=2.4 * stroke,
        zorder=3,
    )
    axes.plot(
        [position],
        [spread.median],
        marker='_',
        markersize=4.0,
        markeredgewidth=1.0,
        color='white',
        zorder=4,
    )
    axes.plot(
        [position],
        [spread.mean],
        marker=MEAN_MARKER,
        markersize=MEAN_SIZE,
        markeredgewidth=0.9,
        markeredgecolor=style.color,
        markerfacecolor='none',
        linestyle='none',
        zorder=5,
    )


def _series(
    frame: pd.DataFrame, models: list[str], column: Column
) -> list[tuple[str, str, ModelStyle] | None]:
    """Which violins a panel of this column holds, in the slots they keep down the whole figure.

    Args:
        frame: The spreads of the panel's log alone, from `read_spreads`.
        models: Every model of the figure, in the order `labels.MODELS` declares them.
        column: The panel's column, which names the log's metric where it has a target.
    Returns:
        One entry per slot: which metric fills it, which model's rows it is read from, and the
        style it is drawn in, or `None` for a slot this log has no run for. The log leads where the
        column names it, being the target the models after it are read against rather than one of
        them.
    """
    scored = [model for model in models if (frame['model'] == model).any()]
    slots: list[tuple[str, str, ModelStyle] | None] = [
        (column.models.key, model, labels.MODELS[model]) if model in scored else None
        for model in models
    ]
    if column.log is None:
        return slots
    # Every model of a log carries the same values for a log's metric, so the first of this log's
    # own says it, and the panel draws one violin rather than one per model that happen to
    # coincide. Read off this log's models rather than the figure's, a log not having to hold all.
    target = (column.log.key, scored[0], labels.LOG_STYLE) if scored else None
    return [target, *slots]


def _draw_panel(axes: Axes, frame: pd.DataFrame, entry: MetricEntry, series: list) -> None:
    """Draw one log's violins for one metric.

    Args:
        axes: The panel to draw onto.
        frame: The spreads of one log alone, from `read_spreads`.
        entry: The metric the panel draws, and what this figure calls it.
        series: Which slot each series occupies, from `_series`.
    """
    for position, slot in enumerate(series):
        if slot is None:
            continue
        key, model, style = slot
        rows = frame[(frame['model'] == model) & (frame['metric'] == key)]
        if rows.empty:
            continue
        _draw_violin(axes, rows.iloc[0]['spread'], position=position, style=style)

    # The metric's own range, exactly as a line of it is drawn against in `compose_figure`: a
    # panel says where a run sits within what the metric can be rather than within the range these
    # runs happen to cover, and either end the metric leaves open is left to the data.
    bottom, top = entry.bounds
    if top is not None:
        top += Y_HEADROOM * (top - (bottom if bottom is not None else 0.0))
    axes.set_ylim(bottom=bottom, top=top)
    # The slots are named once by the legend, so a tick under each of them would say it again a row
    # per log.
    axes.set_xticks([])
    axes.set_xlim(left=-0.5, right=len(series) - 0.5)


def compose_violins(frame: pd.DataFrame, violin: Violin) -> Figure:
    """Compose one violin figure of the catalogue, covering every log at once, untitled since a
    paper captions its figures. A row per log and a column per metric.

    The transpose of `compose_figure`, and the orientation a table reads in: what a panel holds is
    a handful of series rather than a run of lengths, so the logs stack and the metrics run across.

    Args:
        frame: The overall spreads of every run, from `read_spreads`.
        violin: Which figure to draw.
    Returns:
        The finished figure, its rows named by log and its columns titled by metric.
    """
    datasets = labels.DATASETS.ordered(frame['dataset'])
    models = labels.MODELS.ordered(frame['model'])
    columns = violin.columns

    # Page width once there are enough metrics to fill it, and a column of panels of the usual
    # width below that, the same guard `compose_figure` sizes itself by.
    width = min(PAGE_WIDTH, len(columns) * COLUMN_WIDTH)
    figure, grid = plt.subplots(
        nrows=len(datasets),
        ncols=len(columns),
        figsize=(width, len(datasets) * width / len(columns) * ASPECT + LEGEND_HEIGHT),
        squeeze=False,
        constrained_layout=True,
    )
    for panel_row, dataset in zip(grid, datasets, strict=True):
        drawn = frame[frame['dataset'] == dataset]
        for axes, column in zip(panel_row, columns, strict=True):
            _draw_panel(axes, drawn, column.entry, _series(drawn, models, column))
        # Named on the leftmost panel alone, the way a metric names the row it is drawn along in
        # `compose_figure`.
        panel_row[0].set_ylabel(labels.DATASETS[dataset])

    # The metrics title the top row alone: the column below a title is one metric throughout, its
    # unit and which way it reads carried by the label it already has. Wrapped to the width a panel
    # title is, so a long name is not set wider than what it labels.
    for axes, column in zip(grid[0], columns, strict=True):
        axes.set_title(textwrap.fill(column.entry.axis_label, width=TITLE_WIDTH))

    # The violins carry no x ticks, so the legend is the only thing naming them. The log leads it
    # wherever any panel draws one, in the slot it occupies.
    keys = [(labels.MODELS[model].label, labels.MODELS[model].color) for model in models]
    if any(column.log is not None for column in columns):
        keys.insert(0, (labels.LOG_STYLE.label, labels.LOG_STYLE.color))
    legend_above(
        figure,
        [
            Patch(
                facecolor=color,
                alpha=VIOLIN_ALPHA,
                edgecolor=color,
                linewidth=plt.rcParams['axes.linewidth'],
            )
            for _, color in keys
        ],
        [label for label, _ in keys],
    )
    return figure
