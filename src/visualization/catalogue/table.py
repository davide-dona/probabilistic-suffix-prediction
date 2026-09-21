from dataclasses import dataclass

from src.evaluation import Axis
from src.evaluation.scores import METRICS
from src.scalar_metrics import Direction
from src.visualization.catalogue.entry import MetricEntry


@dataclass(frozen=True)
class ColumnGroup:
    """A contiguous group of table columns sharing one header."""

    label: str
    span: int


@dataclass(frozen=True)
class Table:
    """Definition of one comparison table and its columns."""

    name: str
    axis: Axis
    note: str
    columns: tuple[MetricEntry, ...]
    column_groups: tuple[ColumnGroup, ...] = ()

    def __post_init__(self) -> None:
        undirected = [
            entry.metric.key for entry in self.columns if entry.metric.direction is Direction.NONE
        ]
        if undirected:
            raise ValueError(
                f'the {self.name} table holds {", ".join(undirected)}, which have no better value '
                f'and so no cell a reader could rank or the emphasis could mark. A property of the '
                f"log is drawn in FIGURES as the log's own series rather than tabulated."
            )
        if any(group.span < 1 for group in self.column_groups):
            raise ValueError(f'every {self.name} table column group must span at least one column.')
        if self.column_groups and sum(group.span for group in self.column_groups) != len(
            self.columns
        ):
            raise ValueError(
                f'the {self.name} table groups {sum(group.span for group in self.column_groups)} '
                f'columns but declares {len(self.columns)}.'
            )


# Each table answers one evaluation question with directional metrics.
TABLES = (
    # The samples themselves against the one continuation the log took.
    Table(
        name='sample-prediction',
        axis=Axis.OVERALL,
        note="An energy score is E d(X, y) - 0.5 E d(X, X') over the samples, CRPS generalized "
        'off the real line, which charges the spread against the accuracy it buys where the '
        'sampled DL similarity beside it is won by putting every sample on one suffix. Exact is '
        '1 for any '
        'two suffixes that are not the same and bigram is the multiset Jaccard distance over the '
        'ordered activity pairs a suffix holds; both are of negative type, so both are proper. '
        'Normalized Damerau-Levenshtein distance is 1 minus the similarity, and is not of negative '
        'type, so its column is the distance counterpart of the similarity beside it rather than a '
        'proper score.',
        columns=(
            MetricEntry(METRICS['dls_sample_mean'], 'DLS (sample mean)'),
            MetricEntry(METRICS['energy_score_dls'], 'ES (DLD)'),
            MetricEntry(METRICS['energy_score_exact'], 'ES (exact)'),
            MetricEntry(METRICS['energy_score_bigram'], 'ES (bigram)'),
            MetricEntry(METRICS['suffix_length_crps'], 'Suffix length CRPS'),
            MetricEntry(METRICS['suffix_length_mae'], 'Suffix length MAE'),
            MetricEntry(METRICS['inter_event_time_crps_days'], 'Timestamp suffix CRPS'),
            MetricEntry(METRICS['remaining_time_crps_days'], 'Remaining time CRPS'),
        ),
    ),
    # Calibration gaps at three central-interval levels.
    Table(
        name='calibration',
        axis=Axis.OVERALL,
        note="Each cell is the empirical coverage of the samples' central interval less the level "
        'it covers, so 0 is calibrated, negative over-confident and positive over-dispersed. An '
        'interval read off a finite sample is narrow, which costs a calibrated model a point or '
        'two on every column.',
        columns=(
            MetricEntry(METRICS['suffix_length_coverage_gap_50'], r'50\%'),
            MetricEntry(METRICS['suffix_length_coverage_gap_75'], r'75\%'),
            MetricEntry(METRICS['suffix_length_coverage_gap_95'], r'95\%'),
            MetricEntry(METRICS['inter_event_time_coverage_gap_50'], r'50\%'),
            MetricEntry(METRICS['inter_event_time_coverage_gap_75'], r'75\%'),
            MetricEntry(METRICS['inter_event_time_coverage_gap_95'], r'95\%'),
            MetricEntry(METRICS['remaining_time_coverage_gap_50'], r'50\%'),
            MetricEntry(METRICS['remaining_time_coverage_gap_75'], r'75\%'),
            MetricEntry(METRICS['remaining_time_coverage_gap_95'], r'95\%'),
        ),
        column_groups=(
            ColumnGroup('Suffix length', 3),
            ColumnGroup('Timestamp suffix', 3),
            ColumnGroup('Remaining time', 3),
        ),
    ),
)
