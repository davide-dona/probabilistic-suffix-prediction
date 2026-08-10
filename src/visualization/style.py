import logging
from dataclasses import dataclass

import matplotlib as mpl

# Text widths of a two-column paper, in inches. A figure drawn at its final width
# needs no scaling in LaTeX, keeping its font size and line widths correct.
COLUMN_WIDTH = 3.4
PAGE_WIDTH = 7.0
# Height as a share of width, the golden ratio.
DEFAULT_ASPECT = 0.618


@dataclass(frozen=True)
class SeriesStyle:
    """How one run is drawn, the same way in every figure it appears in."""

    color: str
    marker: str
    linestyle: str


# Each style is define by a colour, a marker, and a line style.
# This allows a run to be told apart from the others even when printed in black and white.
SERIES_STYLES = (
    SeriesStyle(color='#0072B2', marker='o', linestyle='-'),
    SeriesStyle(color='#D55E00', marker='s', linestyle='--'),
    SeriesStyle(color='#009E73', marker='^', linestyle='-.'),
    SeriesStyle(color='#7B52AB', marker='D', linestyle=':'),
    SeriesStyle(color='#B08300', marker='v', linestyle=(0, (5, 1, 1, 1, 1, 1))),
    SeriesStyle(color='#CE6E9E', marker='P', linestyle=(0, (3, 1, 3, 1, 1, 1))),
)

# The style every figure shares, matching the LaTeX document style.
# Set once, before any figure is drawn, by `use_paper_style()`.
_PAPER_RC = {
    'font.family': 'serif',
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.titlesize': 9,
    'figure.labelsize': 8,
    # Only the two spines the data is read against.
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.6,
    'axes.labelpad': 3.0,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    # A grid recessive enough to read a value against without competing with the lines.
    'axes.grid': True,
    'axes.axisbelow': True,
    'grid.color': '#000000',
    'grid.alpha': 0.12,
    'grid.linewidth': 0.4,
    'lines.linewidth': 1.2,
    'lines.markersize': 3.0,
    'lines.markeredgewidth': 0.0,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.fancybox': False,
    'legend.edgecolor': '#BBBBBB',
    'legend.facecolor': 'white',
    'legend.borderpad': 0.3,
    'legend.handlelength': 2.2,
    'legend.columnspacing': 1.2,
    'figure.dpi': 200,
    'savefig.dpi': 400,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.01,
    # Embed the text as TrueType rather than as paths, so it stays selectable and searchable in the
    # published PDF.
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
}


def use_paper_style() -> None:
    """Set the shared look every figure shares. Called once, before anything is drawn."""
    # Declare4Py turns the root logger up to DEBUG when it is imported, which makes matplotlib
    # narrate the font subsetting of every figure it writes.
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('fontTools').setLevel(logging.WARNING)
    # Every figure is written to a file and none is shown, so the file backend is the right one
    # whether or not a display happens to be attached.
    mpl.use('Agg')
    mpl.rcParams.update(_PAPER_RC)


def figure_size(width: float, aspect: float = DEFAULT_ASPECT) -> tuple[float, float]:
    """The size of a figure of a given width, in inches.

    Args:
        width: How wide, usually `COLUMN_WIDTH` or `PAGE_WIDTH`.
        aspect: Height as a share of width.
    Returns:
        The width and height to pass as `figsize`.
    """
    return (width, width * aspect)


def series_styles(count: int) -> tuple[SeriesStyle, ...]:
    """The styles for a number of series, assigned in order so a run keeps its look everywhere.

    Args:
        count: How many series will be drawn.
    Returns:
        The first `count` styles.
    Raises:
        ValueError: If more series are asked for than the palette holds.
    """
    if count > len(SERIES_STYLES):
        raise ValueError(
            f'cannot draw {count} series: the palette holds {len(SERIES_STYLES)}, and more lines '
            'than that on one axis cannot be told apart. Visualize fewer runs at a time.'
        )
    return SERIES_STYLES[:count]
