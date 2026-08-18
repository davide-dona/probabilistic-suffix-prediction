"""What the paper draws and tabulates: which figures and which tables, and nothing about how one
is drawn or rendered.

Mirrors `src.visualization.labels`: declared here rather than beside the code that consumes it, so
two runs of the pipeline produce the same page, and a new figure or table is one entry in
`FIGURES` or `TABLES` and nothing else. `Plot` and `Table` read alike, a name, the length
breakdown each is read against and a tuple of `MetricEntry`, though `compose_figure` and
`latex_table` consume them differently: a table holds one breakdown, where a figure draws a row
per metric per breakdown and so names as many as it draws.

A metric names itself nowhere: `src.evaluation.scores.METRICS` declares what each of the numbers
a report carries is, its unit and which way is better, off the score fields themselves, and every
name it goes by on a page is written here as the `MetricEntry` of one appearance. A table holding
one estimator throughout says `Suffix DLS` where a figure drawing that estimator against another
has to say `DLS (point)`, and neither is the metric's own name.
"""

from src.visualization.catalogue.entry import MetricEntry
from src.visualization.catalogue.plot import FIGURES, Plot
from src.visualization.catalogue.table import TABLES, Table

__all__ = [
    'FIGURES',
    'TABLES',
    'MetricEntry',
    'Plot',
    'Table',
]
