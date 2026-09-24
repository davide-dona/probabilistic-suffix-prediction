from dataclasses import dataclass

from src.evaluation import Axis
from src.evaluation.metrics import METRICS
from src.evaluation.metrics.metadata import Metric


@dataclass(frozen=True)
class Plot:
    """Definition of one multi-dataset figure and its panels."""

    name: str
    breakdowns: tuple[Axis, ...]
    panels: tuple[tuple[Metric, ...], ...]


# Catalogue figures: panels by breakdown and dataset.
FIGURES = (
    Plot(
        name='conformance-by-suffix-length',
        breakdowns=(Axis.SUFFIX,),
        panels=((METRICS['conformance_sample_mean'],),),
    ),
)
