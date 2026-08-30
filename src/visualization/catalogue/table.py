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
                f'and so no cell a reader could rank or the emphasis could mark. A spread and a '
                f'property of the log are drawn in VIOLINS rather than tabulated.'
            )


# One table per question a mean can answer. Fidelity and diversity are two of the three goals
# stated for the model; the accuracy table is none of them, but is the comparison against a
# deterministic baseline on the ground the literature already reads it on. Conformance is the
# third goal and has no table: every model of every log sits between 0.93 and 0.99, where a mean
# says nothing a reader can act on, so it is drawn by `spreads` against the log's own conformance
# and by `conformance-by-suffix-length` against the length of what was generated.
#
# A point estimate beats the mean of ten stochastic draws almost by construction, so the two never
# share a column; where a table holds both, each column says which of them it is.
#
# Every column names a metric with a direction. One without has no best value, so its cell is a
# number a reader cannot rank and the emphasis can never mark; those are the log's own values and
# the models' spreads, which `VIOLINS` draws instead.
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
    # EMSC compares the two as stochastic languages; precision asks whether a draw is a
    # continuation the log ever took; the two W1 columns compare the marginals a suffix carries
    # beyond its activities. All four are properties of the distribution a model draws from, which
    # is why the sampled DLS is not among them: it is the expected accuracy of one draw and is
    # maximized by collapsing onto a mode, so it belongs where that collapse is being measured.
    Table(
        name='fidelity',
        axis=Axis.OVERALL,
        note='W1 is the 1-Wasserstein distance; length in events, times in days.',
        columns=(
            MetricEntry(METRICS['emsc'], 'EMSC'),
            MetricEntry(METRICS['continuation_precision'], 'Precision'),
            MetricEntry(METRICS['length_wasserstein'], 'Length W1'),
            MetricEntry(METRICS['remaining_time_wasserstein_days'], 'Rem. time W1'),
        ),
    ),
    # Diversity: whether a model reproduces the spread of continuations the log leaves open, rather
    # than collapsing onto one of them. Recall is the share of what happens the model can produce
    # at all; the mean and the best of a prefix's draws are what that spread buys, one draw against
    # the closest of k, and the distance between the two is the multimodality being useful. What
    # each model's spread actually is, against the log's own, is drawn by `spreads`.
    Table(
        name='diversity',
        axis=Axis.OVERALL,
        note='DLS over k stochastic draws per prefix.',
        columns=(
            MetricEntry(METRICS['continuation_recall'], 'Recall'),
            MetricEntry(METRICS['dls_mean'], 'DLS (mean)'),
            MetricEntry(METRICS['dls_best'], 'DLS (best-of-k)'),
        ),
    ),
)
