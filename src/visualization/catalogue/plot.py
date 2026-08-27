from dataclasses import dataclass

from src.evaluation.report import Axis
from src.evaluation.scores import METRICS
from src.visualization.catalogue.entry import MetricEntry


@dataclass(frozen=True)
class Plot:
    """One figure of the catalogue, covering every log at once: what its file is called, which
    length breakdowns it is drawn against, and the metrics it draws.

    A metric is drawn once per breakdown, so the figure holds one row per pair of the two and one
    column per log. Naming both breakdowns puts them in one figure rather than in two, which is
    worth doing where the two readings say different things and not otherwise: they are not
    independent, since every prefix of a case is scored and so a long prefix leaves a short suffix.
    """

    name: str
    breakdowns: tuple[Axis, ...]
    metrics: tuple[MetricEntry, ...]


# Every figure of the catalogue.
# Each one draws a group of metrics against one or both length breakdowns (PREFIX and SUFFIX).
# One column per log, one row per metric and breakdown, one line per model within a panel.
FIGURES = (
    Plot(
        name='dls-point-by-length',
        breakdowns=(Axis.PREFIX, Axis.SUFFIX),
        metrics=(MetricEntry(METRICS['dls_point'], 'DLS (point)'),),
    ),
    Plot(
        name='dls-mean-by-length',
        breakdowns=(Axis.PREFIX, Axis.SUFFIX),
        metrics=(MetricEntry(METRICS['dls_mean'], 'DLS (sample mean)'),),
    ),
    Plot(
        name='conformance-point-by-suffix-length',
        breakdowns=(Axis.SUFFIX,),
        metrics=(MetricEntry(METRICS['conformance_point'], 'Conformance (point)'),),
    ),
    Plot(
        name='conformance-mean-by-suffix-length',
        breakdowns=(Axis.SUFFIX,),
        metrics=(MetricEntry(METRICS['conformance_mean'], 'Conformance (sample mean)'),),
    ),
    Plot(
        name='remaining-time-point-by-prefix-length',
        breakdowns=(Axis.PREFIX,),
        metrics=(
            MetricEntry(METRICS['remaining_time_ae_point_days'], 'Remaining time MAE (point)'),
        ),
    ),
    Plot(
        name='remaining-time-mean-by-prefix-length',
        breakdowns=(Axis.PREFIX,),
        metrics=(
            MetricEntry(METRICS['remaining_time_ae_mean_days'], 'Remaining time MAE (sample mean)'),
        ),
    ),
    Plot(
        name='time-to-next-point-by-suffix-length',
        breakdowns=(Axis.SUFFIX,),
        metrics=(MetricEntry(METRICS['time_to_next_ae_point_days'], 'Event time MAE (point)'),),
    ),
    Plot(
        name='time-to-next-mean-by-suffix-length',
        breakdowns=(Axis.SUFFIX,),
        metrics=(
            MetricEntry(METRICS['time_to_next_ae_mean_days'], 'Event time MAE (sample mean)'),
        ),
    ),
    Plot(
        name='multi-reference-by-prefix-length',
        breakdowns=(Axis.PREFIX,),
        metrics=(
            MetricEntry(METRICS['reference_energy_score'], 'Energy score (all continuations)'),
            MetricEntry(METRICS['coverage'], 'Coverage'),
            MetricEntry(METRICS['precision'], 'Precision'),
        ),
    ),
    Plot(
        name='marginals-by-prefix-length',
        breakdowns=(Axis.PREFIX,),
        metrics=(
            MetricEntry(METRICS['length_wasserstein'], 'Length W1'),
            MetricEntry(METRICS['remaining_time_wasserstein_days'], 'Remaining time W1'),
        ),
    ),
    Plot(
        name='diversity-by-prefix-length',
        breakdowns=(Axis.PREFIX,),
        metrics=(
            MetricEntry(METRICS['sample_diversity'], 'Sample diversity'),
            MetricEntry(METRICS['unique_sample_rate'], 'Unique sample rate'),
        ),
    ),
)
