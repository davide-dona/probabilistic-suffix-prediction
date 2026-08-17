"""What a model, a dataset and a metric are called, and how a model is drawn.

Cosmetics alone, declared in one place rather than derived from the order files happen to be named,
so two runs of the pipeline produce the same page. Each kind is one `Registry`: `MODELS[name]` is
what is declared for one of them and `MODELS.ordered(names)` sorts a set of them into the order
they are drawn in, both refusing to guess at anything undeclared and naming what to add and where.
A figure is only worth reading if every line in it is labelled the way the paper labels it.

A label with a single consumer stays with it instead: the axis names live beside the figure
catalogue in `figures.py`, and the cloud names beside the distribution grid.
"""

from src.visualization.labels.datasets import DATASETS
from src.visualization.labels.metrics import METRICS
from src.visualization.labels.models import MODELS, reported_models

__all__ = [
    'DATASETS',
    'METRICS',
    'MODELS',
    'reported_models',
]
