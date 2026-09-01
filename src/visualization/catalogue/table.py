from dataclasses import dataclass

from src.evaluation.report import Axis
from src.evaluation.scores import METRICS
from src.scalar_metrics import Direction
from src.visualization.catalogue.entry import MetricEntry


@dataclass(frozen=True)
class Table:
    """One comparison table: what its file is called, which breakdown it summarizes (always the
    overall one, a table comparing runs over their whole split), the sentence its caption has to
    carry, and the columns it holds.

    A table answers one of the questions the paper asks of a model, so the columns are the metrics
    that question is settled on and no others. Rows are a block per log, one per model within it.

    The headers carry no unit. A name and a name plus `[days]` set to different heights in an
    equal-width column, which is what left the header row ragged, so the units are stated once in
    `note` and written into the caption instead of four times over the columns.
    """

    name: str
    axis: Axis
    note: str
    columns: tuple[MetricEntry, ...]

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


# One table per question a mean can answer. Fidelity is one of the two goals stated for the model;
# the accuracy table is neither of them, but is the comparison against a deterministic baseline on
# the ground the literature already reads it on.
#
# Conformance, the other goal, is not tabulated: every model of every log sits within a few points
# of the log's own conformance, where a mean over the whole split says nothing a reader can act on.
# It is drawn by `conformance-by-suffix-length` against that log's own line instead, where the
# lengths and the bands say where a model really parts from the process.
#
# A point estimate beats the mean of ten stochastic draws almost by construction, so the two never
# share a column; where a table holds both, each column says which of them it is.
#
# Every column names a metric with a direction. One without has no best value, so its cell is a
# number a reader cannot rank and the emphasis can never mark; those are the log's own values,
# which `FIGURES` draws as the log's own series, and the diagnostics of a run, which no page draws.
TABLES = (
    # The single suffix each model writes from the mean of `p(z | prefix)`, against the one the log
    # actually took.
    Table(
        name='accuracy-point',
        axis=Axis.OVERALL,
        note='Mean absolute error per prefix; length in events, times in days.',
        columns=(
            MetricEntry(METRICS['dls_point'], 'DLS'),
            MetricEntry(METRICS['length_ae_point'], 'Length'),
            MetricEntry(METRICS['remaining_time_ae_point_days'], 'Rem. time'),
            MetricEntry(METRICS['time_to_next_ae_point_days'], 'Event time'),
        ),
    ),
    # Fidelity: how close the distribution a model draws from is to the one the log continued with.
    # EMSC compares the two as stochastic languages; the three W1 columns compare the marginals
    # a suffix carries beyond its activities. The event-time column groups the waits by the
    # activity they precede, where its counterpart in `accuracy-point` reads them by position:
    # a position only means something once the control flow is right, which EMSC already asks.
    #
    # Precision and recall are the same comparison read in either direction and are deliberately
    # asymmetric, as `scores/distribution.py` sets them out: precision asks whether a draw is a
    # continuation the log ever took, counting a hit whether the log took it once or five hundred
    # times, where recall is weighted by how often each continuation occurred and so is the share
    # of what actually happens that the model reproduces.
    #
    # The sampled DLS is in no table for now. It is the expected accuracy of one draw, which a
    # model collapsed onto a mode maximizes, so it does not settle this question either;
    # `dls-by-length` still draws it against the point estimate.
    Table(
        name='fidelity',
        axis=Axis.OVERALL,
        note='W1 is the 1-Wasserstein distance; length in events, times in days.',
        columns=(
            MetricEntry(METRICS['emsc'], 'EMSC'),
            MetricEntry(METRICS['continuation_precision'], 'Precision'),
            MetricEntry(METRICS['continuation_recall'], 'Recall'),
            MetricEntry(METRICS['length_wasserstein'], 'Length W1'),
            MetricEntry(METRICS['remaining_time_wasserstein_days'], 'Rem. time W1'),
            MetricEntry(METRICS['activity_time_wasserstein_days'], 'Event time W1'),
        ),
    ),
)
