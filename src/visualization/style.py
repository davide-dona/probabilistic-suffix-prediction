import logging
from collections.abc import Sequence

import matplotlib as mpl
from matplotlib.artist import Artist
from matplotlib.figure import Figure

# Text widths of the paper, in inches, matching \documentclass[conference]{IEEEtran} on
# letterpaper: \textwidth 43pc and \columnsep 1pc, neither of which the conference option changes.
# A figure drawn at its final width needs no scaling in LaTeX, keeping its font size and line
# widths correct.
COLUMN_WIDTH = 3.487  # (43pc - 1pc) / 2 = 252pt
PAGE_WIDTH = 7.140  # 43pc = 516pt
# Height as a share of width, the golden ratio.
ASPECT = 0.618

# How much taller than its panels a figure of several of them is, in inches: the room its shared
# legend and its panel titles take above them.
LEGEND_HEIGHT = 0.68
# How much air the layout leaves around the legend, in inches, and so between it and the row of
# titles under it. Wider than the three points a constrained layout leaves by default, at which the
# legend reads as resting on the titles rather than as sitting above them.
LEGEND_PAD = 0.08
# How wide a panel title runs before it wraps onto a second line, and how many intervals a panel's
# x-axis is divided into, both sized for the narrowest panel a group is drawn at.
TITLE_WIDTH = 22
PANEL_X_BINS = 5
# The most markers to draw on one line. Beyond this they merge into the line and stop saying which
# series they belong to.
MAX_MARKERS = 12
# How far past a metric's declared upper bound a panel's y-axis is drawn, as a share of the bound's
# own range. Without it a line approaching the bound reads as clipped against the axis border.
Y_HEADROOM = 0.05

# How much of its series' colour a confidence band carries. Light enough that three of them
# overlapping still read as uncertainty behind the lines rather than as filled areas of their own.
BAND_ALPHA = 0.18
# Where a band and a line sit against each other. Every band of a panel is drawn under every line
# of it, so the last series never covers the first series' line.
BAND_Z = 1.8
LINE_Z = 2.2

# The style every figure shares. Set once, before any figure is drawn, by `apply_style()`.
_PAPER_RC = {
    # The paper sets the body in Times (`\usepackage{times}`); the first of these that resolves
    # on the machine drawing the figure is what it is set in too.
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Times', 'STIX Two Text', 'DejaVu Serif'],
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
    # What a rasterized scatter is written at; the vector parts of a figure are unaffected.
    'savefig.dpi': 400,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.01,
    # Embed the text as TrueType rather than as paths, so it stays selectable and searchable in the
    # published PDF.
    'pdf.fonttype': 42,
}


def apply_style() -> None:
    """Set the shared look every figure shares. Called before anything is drawn."""
    # Declare4Py turns the root logger up to DEBUG when it is imported, which makes matplotlib
    # narrate the font subsetting of every figure it writes and numba the bytecode of every
    # function UMAP compiles.
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('fontTools').setLevel(logging.WARNING)
    logging.getLogger('numba').setLevel(logging.WARNING)
    # Every figure is written to a file and none is shown, so the file backend is the right one
    # whether or not a display happens to be attached.
    mpl.use('Agg')
    mpl.rcParams.update(_PAPER_RC)


def legend_above(figure: Figure, handles: Sequence[Artist], keys: Sequence[str]) -> None:
    """Draw a figure's shared legend in one row above its panels.

    Args:
        figure: The figure, laid out by `constrained_layout` and with its panels already drawn.
        handles: What the legend draws a key for, one per model or per cloud.
        keys: What each of them is called, in the same order.
    """
    figure.legend(handles, keys, loc='outside upper center', ncols=len(keys))
    # The pad the layout leaves between the legend and the row of titles below it. Set here rather
    # than at `subplots`, since it is the legend that needs the room.
    figure.get_layout_engine().set(h_pad=LEGEND_PAD)
