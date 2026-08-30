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
#
# A quantity a model answers both ways is one figure with a row each rather than two files: the
# point estimate and the sample mean are read against each other, and two figures of one quantity
# carry two legends and two sets of lengths to say it.
FIGURES = (
    Plot(
        name='dls-by-length',
        breakdowns=(Axis.PREFIX, Axis.SUFFIX),
        metrics=(
            MetricEntry(METRICS['dls_point'], 'DLS (point)'),
            MetricEntry(METRICS['dls_mean'], 'DLS (sample mean)'),
        ),
    ),
    Plot(
        name='conformance-by-suffix-length',
        breakdowns=(Axis.SUFFIX,),
        metrics=(
            MetricEntry(METRICS['conformance_point'], 'Conformance (point)'),
            MetricEntry(METRICS['conformance_mean'], 'Conformance (sample mean)'),
            MetricEntry(METRICS['conformance_truth'], 'Conformance (ground truth)'),
        ),
    ),
    Plot(
        name='remaining-time-by-prefix-length',
        breakdowns=(Axis.PREFIX,),
        metrics=(
            MetricEntry(METRICS['remaining_time_ae_point_days'], 'Remaining time MAE (point)'),
            MetricEntry(METRICS['remaining_time_ae_mean_days'], 'Remaining time MAE (sample mean)'),
        ),
    ),
    Plot(
        name='time-to-next-by-suffix-length',
        breakdowns=(Axis.SUFFIX,),
        metrics=(
            MetricEntry(METRICS['time_to_next_ae_point_days'], 'Event time MAE (point)'),
            MetricEntry(METRICS['time_to_next_ae_mean_days'], 'Event time MAE (sample mean)'),
        ),
    ),
    Plot(
        name='multi-reference-by-prefix-length',
        breakdowns=(Axis.PREFIX,),
        metrics=(
            MetricEntry(METRICS['emsc'], 'EMSC (all continuations)'),
            MetricEntry(METRICS['continuation_recall'], 'Continuation recall'),
            MetricEntry(METRICS['continuation_precision'], 'Continuation precision'),
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
    # The spread a model draws at against the spread the log's own continuations have, as the
    # prefix grows. The same comparison `spreads` draws over the whole split, read by length.
    Plot(
        name='diversity-by-prefix-length',
        breakdowns=(Axis.PREFIX,),
        metrics=(
            MetricEntry(METRICS['sample_diversity'], 'Sample diversity'),
            MetricEntry(METRICS['reference_diversity'], 'Observed diversity'),
        ),
    ),
)
