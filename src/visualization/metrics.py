from dataclasses import dataclass

from src.evaluation.metrics import PairScores


@dataclass(frozen=True)
class MetricSpec:
    """One of the metrics an evaluation reports, which figures and tables it appears in, and how
    to read it."""

    key: str  # The attribute of `PairScores` that holds it, e.g. `hit_rate_at_1`
    family: str  # Which of `PairScores`' families holds it: `accuracy` or `conformance`
    label: str  # What a figure's axis and a table's Metric column call it
    unit: str | None  # `days` or `events`; `None` for a score already in [0, 1]
    # Defines whether a higher value is better, a lower value is better.
    # `None` for a metric that is not comparable, e.g. a diversity or a length.
    higher_is_better: bool | None


DLS_POINT = MetricSpec(
    key='dls_point',
    family='accuracy',
    label='DL similarity (point)',
    unit=None,
    higher_is_better=True,
)
DLS_MEAN = MetricSpec(
    key='dls_mean',
    family='accuracy',
    label='DL similarity (sample mean)',
    unit=None,
    higher_is_better=True,
)
DLS_BEST = MetricSpec(
    key='dls_best',
    family='accuracy',
    label='DL similarity (best of k)',
    unit=None,
    higher_is_better=True,
)
HIT_RATE_AT_1 = MetricSpec(
    key='hit_rate_at_1',
    family='accuracy',
    label='Hit rate @1',
    unit=None,
    higher_is_better=True,
)
HIT_RATE_AT_5 = MetricSpec(
    key='hit_rate_at_5',
    family='accuracy',
    label='Hit rate @5',
    unit=None,
    higher_is_better=True,
)
HIT_RATE_AT_10 = MetricSpec(
    key='hit_rate_at_10',
    family='accuracy',
    label='Hit rate @10',
    unit=None,
    higher_is_better=True,
)
ENERGY_SCORE = MetricSpec(
    key='energy_score',
    family='accuracy',
    label='Energy score',
    unit=None,
    higher_is_better=False,
)
CONFORMANCE_POINT = MetricSpec(
    key='conformance_point',
    family='conformance',
    label='Conformance (point)',
    unit=None,
    higher_is_better=True,
)
CONFORMANCE_MEAN = MetricSpec(
    key='conformance_mean',
    family='conformance',
    label='Conformance (sample mean)',
    unit=None,
    higher_is_better=True,
)
REMAINING_TIME_AE_POINT = MetricSpec(
    key='remaining_time_ae_point_days',
    family='accuracy',
    label='Remaining time AE (point)',
    unit='days',
    higher_is_better=False,
)
REMAINING_TIME_AE_MEAN = MetricSpec(
    key='remaining_time_ae_mean_days',
    family='accuracy',
    label='Remaining time AE (sample mean)',
    unit='days',
    higher_is_better=False,
)
LENGTH_AE_POINT = MetricSpec(
    key='length_ae_point',
    family='accuracy',
    label='Suffix length AE (point)',
    unit='events',
    higher_is_better=False,
)
LENGTH_AE_MEAN = MetricSpec(
    key='length_ae_mean',
    family='accuracy',
    label='Suffix length AE (sample mean)',
    unit='events',
    higher_is_better=False,
)
SAMPLE_DIVERSITY = MetricSpec(
    key='sample_diversity',
    family='accuracy',
    label='Sample diversity',
    unit=None,
    higher_is_better=None,
)
UNIQUE_SAMPLE_RATE = MetricSpec(
    key='unique_sample_rate',
    family='accuracy',
    label='Unique sample rate',
    unit=None,
    higher_is_better=None,
)
SUFFIX_LENGTH = MetricSpec(
    key='suffix_length',
    family='accuracy',
    label='True suffix length',
    unit='events',
    higher_is_better=None,
)


METRICS = (
    DLS_POINT,
    DLS_MEAN,
    DLS_BEST,
    HIT_RATE_AT_1,
    HIT_RATE_AT_5,
    HIT_RATE_AT_10,
    ENERGY_SCORE,
    CONFORMANCE_POINT,
    CONFORMANCE_MEAN,
    REMAINING_TIME_AE_POINT,
    REMAINING_TIME_AE_MEAN,
    LENGTH_AE_POINT,
    LENGTH_AE_MEAN,
    SAMPLE_DIVERSITY,
    UNIQUE_SAMPLE_RATE,
    SUFFIX_LENGTH,
)

# The metrics that appear in the accuracy
ACCURACY_METRICS = (
    DLS_POINT,
    DLS_MEAN,
    DLS_BEST,
    CONFORMANCE_POINT,
    CONFORMANCE_MEAN,
    HIT_RATE_AT_1,
    HIT_RATE_AT_5,
    HIT_RATE_AT_10,
    ENERGY_SCORE,
)

# What the numeric heads get wrong, and how widely the samples of one prefix scatter: the two
# families read in the log's own units rather than in [0, 1].
ERROR_METRICS = (
    REMAINING_TIME_AE_POINT,
    REMAINING_TIME_AE_MEAN,
    LENGTH_AE_POINT,
    LENGTH_AE_MEAN,
    SAMPLE_DIVERSITY,
    UNIQUE_SAMPLE_RATE,
    SUFFIX_LENGTH,
)


def metric_value(scores: PairScores, spec: MetricSpec) -> float:
    """Read one metric out of a set of scores.

    Args:
        scores: The scores to read, either an evaluation's overall means or one length's.
        spec: Which metric to read.
    Returns:
        The metric's value.
    """
    return getattr(getattr(scores, spec.family), spec.key)


def axis_label(spec: MetricSpec) -> str:
    """A metric's axis label, its unit appended where it has one."""
    return spec.label if spec.unit is None else f'{spec.label} [{spec.unit}]'


def format_value(value: float, spec: MetricSpec) -> str:
    """Format a metric for a table cell.

    Args:
        value: The value to format.
        spec: The metric it belongs to, whose unit decides the precision: three decimals for a
            score already in [0, 1], two for one carrying a unit and so a wider range.
    Returns:
        The formatted value.
    """
    return f'{value:.3f}' if spec.unit is None else f'{value:.2f}'
