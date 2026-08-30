from dataclasses import dataclass

from src.evaluation.report import Axis
from src.evaluation.scores import METRICS
from src.scalar_metrics import Metric
from src.visualization.catalogue.entry import MetricEntry


@dataclass(frozen=True)
class Column:
    """One column of a comparison table: what fills its model rows, what fills the log's own row
    under it, and what the column is headed.

    At least one of the two must be given. A column with `models` alone is a plain comparison; one
    with `log` alone is a property of the log the models are read against, e.g. how many
    continuations it was observed to take; one with both puts every model beside the target it is
    chasing, which is what a gap column's emphasis is read against.
    """

    label: str
    models: Metric | None = None
    log: Metric | None = None

    def __post_init__(self) -> None:
        if self.models is None and self.log is None:
            raise ValueError(
                f'the {self.label} column names neither a metric for the models nor one for the '
                f'log, so nothing would be written under it.'
            )

    @property
    def entry(self) -> MetricEntry:
        """How the column is headed and how its cells are written, from whichever metric it names.

        The two metrics of a column are the same number measured on the model and on the log, so
        they share a unit and a format and either can say how the column reads.
        """
        metric = self.models if self.models is not None else self.log
        assert metric is not None
        return MetricEntry(metric, self.label)


@dataclass(frozen=True)
class Table:
    """One comparison table: what its file is called, which breakdown it summarizes (always the
    overall one, a table comparing runs over their whole split), and the columns it holds.

    A table answers one of the questions the paper asks of a model, so the columns are the metrics
    that question is settled on and no others. Rows are a block per log, one per model within it,
    with the log's own row above them wherever a column declares a target.
    """

    name: str
    axis: Axis
    columns: tuple[Column, ...]

    @property
    def has_log_row(self) -> bool:
        """Whether any column of the table carries a value for the log's own row."""
        return any(column.log is not None for column in self.columns)


# One table per question the paper asks. Fidelity, diversity and conformance are the three goals
# stated for the model; the accuracy table is none of them, but is the comparison against a
# deterministic baseline on the ground the literature already reads it on.
#
# A point estimate beats the mean of ten stochastic draws almost by construction, so the two never
# share a column; where a table holds both, each column says which of them it is.
TABLES = (
    # The single suffix each model writes from the mean of `p(z | prefix)`, against the one the log
    # actually took.
    Table(
        name='accuracy-point',
        axis=Axis.OVERALL,
        columns=(
            Column('DLS', models=METRICS['dls_point']),
            Column('Length MAE', models=METRICS['length_ae_point']),
            Column('Remaining time MAE', models=METRICS['remaining_time_ae_point_days']),
            Column('Event time MAE', models=METRICS['time_to_next_ae_point_days']),
        ),
    ),
    # Fidelity: how close the distribution a model draws from is to the one the log continued with.
    # EMSC compares the two as stochastic languages; precision asks whether a draw is a
    # continuation the log ever took; the two W1 columns compare the marginals a suffix carries
    # beyond its activities.
    Table(
        name='fidelity',
        axis=Axis.OVERALL,
        columns=(
            Column('EMSC', models=METRICS['emsc']),
            Column('Precision', models=METRICS['continuation_precision']),
            Column('DLS', models=METRICS['dls_mean']),
            Column('Length W1', models=METRICS['length_wasserstein']),
            Column('Remaining time W1', models=METRICS['remaining_time_wasserstein_days']),
        ),
    ),
    # Diversity: whether a model reproduces the spread of continuations the log leaves open, rather
    # than collapsing onto one of them. The gap is what is emphasized, since spreading wider than
    # the process does is as wrong as collapsing; recall and best-of-k DLS are what that spread
    # buys, the share of what happens the model can produce at all.
    Table(
        name='diversity',
        axis=Axis.OVERALL,
        columns=(
            Column(
                'Diversity',
                models=METRICS['sample_diversity'],
                log=METRICS['reference_diversity'],
            ),
            Column('Diversity gap', models=METRICS['diversity_gap']),
            Column('Unique draws', models=METRICS['unique_sample_rate']),
            Column('Recall', models=METRICS['continuation_recall']),
            Column('Best-of-k DLS', models=METRICS['dls_best']),
            Column('Observed continuations', log=METRICS['reference_size']),
        ),
    ),
    # Conformance: whether generated suffixes obey the constraints of the process, and whether they
    # obey them more closely than the traces the process itself produced. The level is what a
    # reader wants to see and the gap is what is emphasized, a level near 1.0 being what a model
    # emitting only the log's commonest variant would also score.
    Table(
        name='conformance',
        axis=Axis.OVERALL,
        columns=(
            Column(
                'Conformance (sampled)',
                models=METRICS['conformance_mean'],
                log=METRICS['conformance_truth'],
            ),
            Column('Gap (sampled)', models=METRICS['conformance_gap_mean']),
            Column(
                'Conformance (point)',
                models=METRICS['conformance_point'],
                log=METRICS['conformance_truth'],
            ),
            Column('Gap (point)', models=METRICS['conformance_gap_point']),
        ),
    ),
)
