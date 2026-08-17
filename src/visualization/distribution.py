from dataclasses import dataclass, replace

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

from src.visualization import labels
from src.visualization.embedding import Source
from src.visualization.style import PAGE_WIDTH, legend_above

# The fewest points a density can be estimated from at all. A cloud of a handful of distinct
# suffixes has no density worth drawing.
MIN_CONTOUR_POINTS = 20


@dataclass(frozen=True)
class CloudStyle:
    """How one cloud of suffixes is drawn: its points, its outline and the wash under it.

    What sits above what within a panel is set by the layers rather than by the order the clouds
    are drawn in, so the truth's outline stays readable under a model's points.
    """

    color: str
    point_size: float
    point_alpha: float
    point_layer: int
    # Dashed for the truth against solid for the model, so the two outlines are told apart wherever
    # they cross and without colour.
    linestyle: str | tuple[int, tuple[int, ...]]
    contour_layer: int
    # The layer to wash the outermost contour in at, or `None` to draw lines alone.
    fill_layer: int | None


# The ground truth is the reference every panel is read against, so it is drawn in a grey recessive
# enough not to compete with a model's colour and still be visible under it. It is the field a
# model's cloud is drawn onto rather than a second cloud competing with it, so its points are
# smaller and fainter and its shape is carried by the contour and the wash under it, which is also
# why it is the only one of the two that is filled.
TRUTH_CLOUD = CloudStyle(
    color='#8A8A8A',
    point_size=1.8,
    point_alpha=0.18,
    point_layer=2,
    linestyle=(0, (4, 2)),
    contour_layer=3,
    fill_layer=1,
)
# Drawn in the colour of the model whose panel it is, which `_draw_panel` replaces this one with.
GENERATED_CLOUD = CloudStyle(
    color='#3A3A3A',
    point_size=2.4,
    point_alpha=0.5,
    point_layer=4,
    linestyle='-',
    contour_layer=5,
    fill_layer=None,
)

TRUTH_LABEL = 'Ground truth'
GENERATED_LABEL = 'Generated'
# What the legend's generated key is drawn in. The colour of that cloud belongs to the panel and is
# already its title, so the key can only carry the line style every panel shares.
LEGEND_COLOR = '#3A3A3A'

# The share of a cloud's mass each contour line encloses, from its core outwards. The same levels
# for both clouds, so the two outlines mean the same thing.
CONTOUR_LEVELS = (0.25, 0.5, 0.75)
CONTOUR_WIDTH = 0.7
# How far a contour's white halo reaches past the line itself. Without it a line is drawn in the
# colour of the points it crosses, and a tight cloud's outline disappears into its own core.
CONTOUR_HALO_WIDTH = 0.7
# How fine the grid a contour is traced on is.
CONTOUR_RESOLUTION = 120
# How solid the wash under the ground truth's outermost contour is. Enough to find the reference
# under a model's cloud, light enough to read that cloud's own points through.
FILL_ALPHA = 0.16
# How far past the outermost point the axes reach, as a share of the embedding's extent.
AXIS_MARGIN = 0.04
# How tall a row of panels is beyond the panels themselves, in inches: room for a title and the
# legend above them.
HEADER_HEIGHT = 0.52
# How tall a panel may be relative to its width. Both axes are drawn to one scale, so a row's
# height follows the shape of the embedding itself; these bound how flat, or how narrow, an
# embedding of an unusual shape may leave it.
MIN_PANEL_ASPECT = 0.55
MAX_PANEL_ASPECT = 1.1

# The (x, y) ranges every panel of a figure is drawn over.
type Limits = tuple[tuple[float, float], tuple[float, float]]


def _points(frame: pd.DataFrame) -> np.ndarray:
    """The coordinates of a selection of the embedding. `[num_points, 2]`"""
    return frame[['x', 'y']].to_numpy()


def _limits(points: np.ndarray) -> Limits:
    """The axis ranges holding every point of a figure, with a margin past the outermost."""
    low, high = points.min(axis=0), points.max(axis=0)  # [2] each
    margin = AXIS_MARGIN * np.maximum(high - low, 1e-6)  # [2]
    low, high = low - margin, high + margin
    return ((float(low[0]), float(high[0])), (float(low[1]), float(high[1])))


def _spread(points: np.ndarray) -> float:
    """How wide a set of coordinates is, on the scale `gaussian_kde` sizes its kernel by: the
    fourth root of their covariance determinant, or 0.0 for a cloud with no area to speak of."""
    determinant = float(np.linalg.det(np.cov(points.T)))
    return determinant**0.25 if np.isfinite(determinant) and determinant > 0.0 else 0.0


def _draw_contours(
    axes: Axes,
    points: np.ndarray,
    cloud: CloudStyle,
    limits: Limits,
    smoothing: float,
) -> None:
    """Outline where a cloud's mass sits, over the scatter that shows its individual suffixes."""
    if len(points) < MIN_CONTOUR_POINTS or len(np.unique(points, axis=0)) < 3:
        return
    spread = _spread(points)
    if spread == 0.0 or smoothing == 0.0:
        # A cloud spread over a line has no two-dimensional density to estimate.
        return
    try:
        # The factor scales this cloud's own covariance to the width the whole figure is drawn at.
        kernel = gaussian_kde(points.T, bw_method=smoothing / spread)
    except (np.linalg.LinAlgError, ValueError):
        return

    # The level enclosing a share of the mass is the density at the point that share of the cloud,
    # taken from the densest outwards, sits above.
    at_points = np.sort(kernel(points.T))[::-1]  # [num_points]
    share = np.cumsum(at_points) / at_points.sum()  # [num_points]
    levels = sorted(
        {
            float(at_points[min(int(np.searchsorted(share, level)), len(at_points) - 1)])
            for level in CONTOUR_LEVELS
        }
    )

    (left, right), (bottom, top) = limits
    x_grid, y_grid = np.meshgrid(
        np.linspace(start=left, stop=right, num=CONTOUR_RESOLUTION),
        np.linspace(start=bottom, stop=top, num=CONTOUR_RESOLUTION),
    )  # [resolution, resolution] each
    density = kernel(np.vstack([x_grid.ravel(), y_grid.ravel()])).reshape(x_grid.shape)

    if cloud.fill_layer is not None:
        # One wash over everything above the outermost level, rather than one per band: a stack of
        # translucent bands reads as a gradient and hides the model's points under it.
        axes.contourf(
            x_grid,
            y_grid,
            density,
            levels=[levels[0], float(density.max())],
            colors=[cloud.color],
            alpha=FILL_ALPHA,
            zorder=cloud.fill_layer,
        )
    contours = axes.contour(
        x_grid,
        y_grid,
        density,
        levels=levels,
        colors=cloud.color,
        linewidths=CONTOUR_WIDTH,
        linestyles=[cloud.linestyle],
        zorder=cloud.contour_layer,
    )
    # A halo, so a line stays readable where it crosses the densest part of its own cloud and is
    # otherwise drawn in the colour of the points under it.
    contours.set_path_effects(
        [path_effects.withStroke(linewidth=CONTOUR_WIDTH + CONTOUR_HALO_WIDTH, foreground='white')]
    )


def _draw_panel(
    axes: Axes,
    truth: np.ndarray,
    generated: np.ndarray,
    color: str,
    limits: Limits,
    smoothing: float,
) -> None:
    """Draw one model's cloud over the ground truth's, on one set of axes.

    What sits above what is set by each cloud's layers rather than by the order they are drawn in,
    so the truth's outline stays readable under the model's points instead of being covered by them.
    """
    clouds = (
        (truth, TRUTH_CLOUD),
        (generated, replace(GENERATED_CLOUD, color=color)),
    )
    for points, cloud in clouds:
        # Rasterized: a panel holds thousands of points, and as vectors they would dominate the
        # file while the contours over them stay sharp either way.
        axes.scatter(
            points[:, 0],
            points[:, 1],
            s=cloud.point_size,
            c=cloud.color,
            alpha=cloud.point_alpha,
            linewidths=0.0,
            rasterized=True,
            zorder=cloud.point_layer,
        )
        _draw_contours(axes=axes, points=points, cloud=cloud, limits=limits, smoothing=smoothing)

    axes.set_xlim(*limits[0])
    axes.set_ylim(*limits[1])
    # Both axes span the same range, so drawing them to the same scale costs no whitespace and
    # keeps a cluster's shape the embedding's rather than the panel box's.
    axes.set_aspect('equal', adjustable='box')
    # A UMAP coordinate has no unit and no reading, so nothing about the axes is worth printing.
    axes.set_xticks([])
    axes.set_yticks([])
    axes.grid(visible=False)
    for spine in axes.spines.values():
        spine.set_visible(True)


def distribution_grid(frame: pd.DataFrame) -> Figure:
    """A page-wide row of what every model of one log generates, against the ground truth.

    Args:
        frame: The embedded suffixes of one log, from `embed_suffixes`. Every model of it shares
            one embedding, so the panels are read against each other as well as against the truth.
    Returns:
        The finished figure, its panels titled by model.
    """
    models = labels.ordered_models(frame['model'])
    # Every point the figure is drawn from, the ground truth counted once rather than once per
    # panel it is repeated under.
    generated = frame['source'] == Source.GENERATED
    truth = (frame['source'] == Source.TRUTH) & (frame['model'] == models[0])
    pooled = _points(frame[generated | truth])

    limits = _limits(pooled)
    # `gaussian_kde` sizes its kernel by the covariance of the cloud it is handed, so a tight cloud
    # would be smoothed less than a broad one and their outlines would not be on the same footing.
    # One width taken from every point of the figure at once, Scott's rule in two dimensions times
    # the spread it is a share of, is what lets the panels be read against each other the way the
    # shared embedding lets their positions be.
    smoothing = len(pooled) ** (-1.0 / 6.0) * _spread(pooled)

    # A panel is drawn to one scale on both axes, so the row is as tall, against how wide it is,
    # as the embedding it holds. Sizing the figure to that is what keeps the equal scale from
    # paying for itself in empty bands above and below the clouds.
    (left, right), (bottom, top) = limits
    aspect = min(max((top - bottom) / (right - left), MIN_PANEL_ASPECT), MAX_PANEL_ASPECT)
    panel_width = PAGE_WIDTH / len(models)

    figure, grid = plt.subplots(
        nrows=1,
        ncols=len(models),
        figsize=(PAGE_WIDTH, panel_width * aspect + HEADER_HEIGHT),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    for axes, model in zip(grid[0], models, strict=True):
        panel = frame[frame['model'] == model]
        model_style = labels.model_style(model)
        _draw_panel(
            axes=axes,
            truth=_points(panel[panel['source'] == Source.TRUTH]),
            generated=_points(panel[panel['source'] == Source.GENERATED]),
            color=model_style.color,
            limits=limits,
            smoothing=smoothing,
        )
        # In the model's own colour, so a title names the cloud it belongs to as well as the run.
        axes.set_title(model_style.label, color=model_style.color)

    # Two keys carrying the line styles alone, since what tells the two clouds apart in a panel is
    # the dashing: the truth keeps one colour throughout while the generated cloud takes the
    # panel's, which is already named by its title, so its key is drawn in a neutral instead.
    legend_keys = (
        (TRUTH_CLOUD.color, TRUTH_CLOUD.linestyle, TRUTH_LABEL),
        (LEGEND_COLOR, GENERATED_CLOUD.linestyle, GENERATED_LABEL),
    )
    legend_above(
        figure,
        [
            Line2D([], [], color=color, linestyle=linestyle, marker='o', markersize=3.0)
            for color, linestyle, _ in legend_keys
        ],
        [label for _, _, label in legend_keys],
    )
    return figure
