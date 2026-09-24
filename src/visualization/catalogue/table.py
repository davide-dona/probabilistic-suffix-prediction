from dataclasses import dataclass

from src.evaluation import Axis
from src.evaluation.metrics import METRICS
from src.evaluation.metrics.metadata import Direction, Metric


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
    columns: tuple[Metric, ...]
    column_groups: tuple[ColumnGroup, ...] = ()

    def __post_init__(self) -> None:
        undirected = [metric.key for metric in self.columns if metric.direction is Direction.NONE]
        if undirected:
            raise ValueError(f'{self.name} has metrics without a ranking direction: {undirected}')
        if any(group.span < 1 for group in self.column_groups):
            raise ValueError(f'every {self.name} table column group must span at least one column.')
        if self.column_groups and sum(group.span for group in self.column_groups) != len(
            self.columns
        ):
            raise ValueError(f'{self.name} column groups do not cover its columns.')


# Each table answers one evaluation question with directional metrics.
TABLES = (
    Table(
        name='sample-prediction',
        axis=Axis.OVERALL,
        columns=(
            METRICS['dls_sample_mean'],
            METRICS['energy_score_dls'],
            METRICS['energy_score_exact'],
            METRICS['energy_score_bigram'],
            METRICS['suffix_length_mae'],
            METRICS['suffix_length_crps'],
        ),
    ),
    Table(
        name='calibration',
        axis=Axis.OVERALL,
        columns=(
            METRICS['suffix_length_coverage_gap_50'],
            METRICS['suffix_length_coverage_gap_75'],
            METRICS['suffix_length_coverage_gap_95'],
        ),
        column_groups=(ColumnGroup('Suffix length', 3),),
    ),
)
