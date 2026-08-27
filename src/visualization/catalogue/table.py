from dataclasses import dataclass

from src.evaluation.report import Axis
from src.evaluation.scores import METRICS
from src.visualization.catalogue.entry import MetricEntry


@dataclass(frozen=True)
class Table:
    """One comparison table: what its file is called, which breakdown it summarizes (always the
    overall one, a table comparing runs over their whole split), and the metrics it puts in rows,
    each under the name its single estimator lets it be short about."""

    name: str
    axis: Axis
    metrics: tuple[MetricEntry, ...]


# Each table answers one question, and each holds a single estimator so that the emphasis compares
# models rather than a model against itself. A point estimate beats the mean of ten stochastic
# draws almost by construction, so the two must never share a row.
TABLES = (
    # Point prediction results
    Table(
        name='comparison-point',
        axis=Axis.OVERALL,
        metrics=(
            MetricEntry(METRICS['dls_point'], 'DLS'),
            MetricEntry(METRICS['conformance_point'], 'Conformance'),
            MetricEntry(METRICS['length_ae_point'], 'Length MAE'),
            MetricEntry(METRICS['remaining_time_ae_point_days'], 'Remaining time MAE'),
            MetricEntry(METRICS['time_to_next_ae_point_days'], 'Event time MAE'),
        ),
    ),
    # Probabilistic generation results
    Table(
        name='comparison-probabilistic',
        axis=Axis.OVERALL,
        metrics=(
            MetricEntry(METRICS['dls_mean'], 'DLS'),
            MetricEntry(METRICS['conformance_mean'], 'Conformance'),
            MetricEntry(METRICS['length_ae_mean'], 'Length MAE'),
            MetricEntry(METRICS['remaining_time_ae_mean_days'], 'Remaining time MAE'),
            MetricEntry(METRICS['time_to_next_ae_mean_days'], 'Event time MAE'),
        ),
    ),
    # Comparison of distributions, against the one continuation each prefix actually took
    Table(
        name='comparison-distribution',
        axis=Axis.OVERALL,
        metrics=(
            MetricEntry(METRICS['energy_score'], 'Energy score'),
            MetricEntry(METRICS['hit_rate_at_1'], 'Hit rate @1'),
            MetricEntry(METRICS['hit_rate_at_5'], 'Hit rate @5'),
            MetricEntry(METRICS['hit_rate_at_10'], 'Hit rate @10'),
            MetricEntry(METRICS['dls_best'], 'Best-of-k DLS'),
        ),
    ),
    # Comparison of distributions, against every continuation the log took after the prefix
    Table(
        name='comparison-multi-reference',
        axis=Axis.OVERALL,
        metrics=(
            MetricEntry(METRICS['reference_energy_score'], 'Energy score (all continuations)'),
            MetricEntry(METRICS['coverage'], 'Coverage'),
            MetricEntry(METRICS['precision'], 'Precision'),
            MetricEntry(METRICS['length_wasserstein'], 'Length W1'),
            MetricEntry(METRICS['remaining_time_wasserstein_days'], 'Remaining time W1'),
            MetricEntry(METRICS['reference_size'], 'Observed continuations'),
        ),
    ),
)
