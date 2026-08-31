from dataclasses import dataclass

from src.evaluation.scores import METRICS
from src.scalar_metrics import Metric, Owner
from src.visualization.catalogue.entry import MetricEntry


@dataclass(frozen=True)
class Column:
    """One panel column of a violin figure: what the models' violins are drawn from, what the log's
    own violin beside them is drawn from, and what the column is headed.

    A column with `models` alone compares the models against each other; one that also names a
    `log` metric puts every model beside the target it is chasing, which is the whole point of
    drawing the log as a series rather than writing it as a row.
    """

    label: str
    models: Metric
    log: Metric | None = None

    def __post_init__(self) -> None:
        if self.models.owner is not Owner.MODEL:
            raise ValueError(
                f'the {self.label} column draws {self.models.key} as its models, but that metric '
                f'is the log’s own, so every model would draw the same shape.'
            )
        if self.log is not None and self.log.owner is not Owner.LOG:
            raise ValueError(
                f'the {self.label} column draws {self.log.key} as the log, but that metric is a '
                f'model’s own, so it has no single value the log could be drawn at.'
            )

    @property
    def entry(self) -> MetricEntry:
        """How the column is headed and how its values read, from the metric its models draw.

        The two metrics of a column are the same number measured on the model and on the log, so
        they share a unit and either can say how the column reads.
        """
        return MetricEntry(self.models, self.label)


@dataclass(frozen=True)
class Violin:
    """One violin figure: what its file is called, and the metrics it draws.

    A row per log and a column per metric, the transpose of `Plot` and the orientation a table
    reads in, since what a panel holds is a handful of categories rather than a run of lengths.
    Within a panel, one violin per model plus the log's own wherever the column names a target.
    """

    name: str
    columns: tuple[Column, ...]


# Every violin figure of the catalogue. These are the questions a mean cannot answer: what a
# model's spread over the split actually looks like, and where the log's own value falls among the
# models rather than beside them in a table. A metric with no direction belongs here rather than in
# a table, having no best value a cell could be ranked on.
#
# One figure, since every column asks the same question of a different metric: where each model
# sits against the log it is read against. That comparison is drawn once here rather than restated
# as a gap column beside the two levels it is the difference of.
VIOLINS = (
    Violin(
        name='spreads',
        columns=(
            # How far apart a model spreads a prefix's draws, against how far apart the log's own
            # continuations of it are. The log is the target rather than a competitor.
            Column(
                'Diversity',
                models=METRICS['sample_diversity'],
                log=METRICS['reference_diversity'],
            ),
            # How far each model's suffixes obey the process, against how far the ones the process
            # itself produced do. The log is drawn in both panels: each is a comparison of its own,
            # where a table writing the same number twice only read as an error. Named the short
            # way round, so every panel title of a row is one line and the three sit level.
            Column(
                'Sampled conformance',
                models=METRICS['conformance_mean'],
                log=METRICS['conformance_truth'],
            ),
            Column(
                'Point conformance',
                models=METRICS['conformance_point'],
                log=METRICS['conformance_truth'],
            ),
        ),
    ),
)
