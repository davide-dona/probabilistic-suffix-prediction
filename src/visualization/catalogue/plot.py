from dataclasses import dataclass

from src.evaluation import Axis
from src.evaluation.scores import METRICS
from src.visualization.catalogue.entry import MetricEntry


@dataclass(frozen=True)
class Plot:
    """Definition of one multi-dataset figure and its panels."""

    name: str
    breakdowns: tuple[Axis, ...]
    panels: tuple[tuple[MetricEntry, ...], ...]


# Catalogue figures: panels by breakdown and dataset.
FIGURES = (
    Plot(
        name='dls-by-suffix-length',
        breakdowns=(Axis.SUFFIX,),
        panels=(
            (MetricEntry(METRICS['dls_point'], 'DLS (point)'),),
            (MetricEntry(METRICS['dls_sample_mean'], 'DLS (sample mean)'),),
        ),
    ),
    Plot(
        name='conformance-by-suffix-length',
        breakdowns=(Axis.SUFFIX,),
        panels=(
            (
                MetricEntry(METRICS['conformance_point'], 'Conformance (point)'),
                MetricEntry(METRICS['conformance_observed'], 'Conformance (observed)'),
            ),
            (
                MetricEntry(METRICS['conformance_sample_mean'], 'Conformance (sample mean)'),
                MetricEntry(METRICS['conformance_observed'], 'Conformance (observed)'),
            ),
            (
                MetricEntry(
                    METRICS['full_conformance_sample_rate'], 'Fully conformant (samples)'
                ),
                MetricEntry(
                    METRICS['full_conformance_observed'], 'Fully conformant (observed)'
                ),
            ),
        ),
    ),
)
