from dataclasses import dataclass

from src.evaluation import Axis
from src.evaluation.metrics import METRICS
from src.evaluation.metrics.definitions import Direction
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
    note: str | None
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
    Table(
        name='sample-prediction',
        axis=Axis.OVERALL,
        note=None,
        columns=(
            MetricEntry(METRICS['dls_sample_mean'], 'DLS mean'),
            MetricEntry(METRICS['energy_score_dls'], r'$ES_{\mathrm{DL}}$'),
            MetricEntry(METRICS['energy_score_exact'], r'$ES_{\mathrm{exact}}$'),
            MetricEntry(METRICS['energy_score_bigram'], r'$ES_{\mathrm{2-\text{gram}}}$'),
            MetricEntry(METRICS['suffix_length_mae'], 'Suffix length mean'),
            MetricEntry(METRICS['suffix_length_crps'], 'Suffix length CRPS'),
        ),
    ),
    Table(
        name='calibration',
        axis=Axis.OVERALL,
        note=None,
        columns=(
            MetricEntry(METRICS['suffix_length_coverage_gap_50'], r'50\%'),
            MetricEntry(METRICS['suffix_length_coverage_gap_75'], r'75\%'),
            MetricEntry(METRICS['suffix_length_coverage_gap_95'], r'95\%'),
        ),
        column_groups=(ColumnGroup('Suffix length', 3),),
    ),
)
