"""What the paper draws and tabulates: which figures and which tables, and nothing about how one
is drawn or rendered.

Mirrors `src.visualization.labels`: declared here rather than beside the code that consumes it, so
two runs of the pipeline produce the same page, and a new figure or table is one entry in
`FIGURES` or `TABLES` and nothing else. `Plot` and `Table` read alike, a name, the length
breakdown each is read against and a tuple of appearances, though `compose_figure` and
`latex_table` consume them differently: a table holds one breakdown and lays its appearances out
as columns, where a figure draws a row per metric per breakdown and so names as many breakdowns as
it draws. A table's appearance is a `Column` rather than a bare `MetricEntry`, since a column can
name two metrics: what each model scored, and what the log the models are read against did.

A metric names itself nowhere: `src.evaluation.scores.METRICS` declares what each of the numbers
a report carries is, its unit and which way it reads, off the score fields themselves, and every
name it goes by on a page is written here as one appearance. A table naming a single estimator
says `DLS` where a figure drawing that estimator against another has to say `DLS (point)`, and
neither is the metric's own name.
"""

from src.visualization.catalogue.entry import MetricEntry
from src.visualization.catalogue.plot import FIGURES, Plot
from src.visualization.catalogue.table import TABLES, Column, Table

__all__ = [
    'FIGURES',
    'TABLES',
    'Column',
    'MetricEntry',
    'Plot',
    'Table',
]
