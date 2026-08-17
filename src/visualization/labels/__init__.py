"""What a model, a dataset and a metric are called, and how a model is drawn.

Cosmetics alone, declared in one place rather than derived from the order files happen to be named,
so two runs of the pipeline produce the same page. Nothing declared for an entry stops the run,
naming what to add and where: a figure is only worth reading if every line in it is labelled the
way the paper labels it.

A label with a single consumer stays with it instead: the axis names live beside the figure
catalogue in `figures.py`, and the cloud names beside the distribution grid.
"""

from src.visualization.labels.datasets import DATASETS, dataset_label, ordered_datasets
from src.visualization.labels.metrics import METRICS, Metric, metric
from src.visualization.labels.models import (
    MODELS,
    ModelStyle,
    model_style,
    ordered_models,
    reported,
    reported_models,
)

__all__ = [
    'DATASETS',
    'METRICS',
    'MODELS',
    'Metric',
    'ModelStyle',
    'dataset_label',
    'metric',
    'model_style',
    'ordered_datasets',
    'ordered_models',
    'reported',
    'reported_models',
]
