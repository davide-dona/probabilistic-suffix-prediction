"""How one figure or one table is made, and nothing about which ones a run writes.

`compose_figure`, `distribution_grid` and `latex_table` each take a tidy frame and return a
`Figure` or a string; `pipelines.visualize` is what walks the logs, names the files and writes them.

`embed_suffixes` is the one entry point that reads files rather than a frame. It lays a set of
generations out with UMAP, which is a choice about what makes a legible page rather than about how
a model is scored, so it lives beside the figure that reads it instead of in `src.evaluation`. The
frame every other figure is drawn from comes from `src.evaluation.report.read_reports`, which is
the read half of a format the evaluation pipeline writes.
"""

from src.visualization.distribution import distribution_grid
from src.visualization.embedding import embed_suffixes
from src.visualization.figures import FIGURES, compose_figure
from src.visualization.labels import reported_models
from src.visualization.style import apply_style
from src.visualization.tables import TABLES, latex_table

__all__ = [
    'FIGURES',
    'TABLES',
    'apply_style',
    'compose_figure',
    'distribution_grid',
    'embed_suffixes',
    'latex_table',
    'reported_models',
]
