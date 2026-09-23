from src.evaluation.metrics.definitions import Direction, MetricGroup, Unit
from src.evaluation.metrics.helpers.statistics import coverage_gap, crps, mae
from src.evaluation.metrics.prepared import PreparedPrefix
from src.evaluation.metrics.registry import METRICS


@METRICS.register(
    'suffix_length_mae',
    label='Suffix length MAE',
    group=MetricGroup.SUFFIX_LENGTH,
    unit=Unit.EVENTS,
    direction=Direction.LOWER,
)
def suffix_length_mae(context: PreparedPrefix) -> float:
    """Return the sampled suffix-length mean absolute error."""
    return mae(context.suffix_lengths, context.true_suffix_length)


@METRICS.register(
    'suffix_length_crps',
    label='Suffix length CRPS',
    group=MetricGroup.SUFFIX_LENGTH,
    unit=Unit.EVENTS,
    direction=Direction.LOWER,
)
def suffix_length_crps(context: PreparedPrefix) -> float:
    """Return the sampled suffix-length CRPS."""
    return crps(context.suffix_lengths, context.true_suffix_length)


@METRICS.register(
    'suffix_length_coverage_gap_50',
    label='Suffix length coverage gap 50%',
    group=MetricGroup.SUFFIX_LENGTH,
    unit=Unit.SCORE,
    direction=Direction.ZERO,
)
def suffix_length_coverage_gap_50(context: PreparedPrefix) -> float:
    """Return the 50% suffix-length central-interval coverage gap."""
    return coverage_gap(context.suffix_lengths, context.true_suffix_length, level=0.50)


@METRICS.register(
    'suffix_length_coverage_gap_75',
    label='Suffix length coverage gap 75%',
    group=MetricGroup.SUFFIX_LENGTH,
    unit=Unit.SCORE,
    direction=Direction.ZERO,
)
def suffix_length_coverage_gap_75(context: PreparedPrefix) -> float:
    """Return the 75% suffix-length central-interval coverage gap."""
    return coverage_gap(context.suffix_lengths, context.true_suffix_length, level=0.75)


@METRICS.register(
    'suffix_length_coverage_gap_95',
    label='Suffix length coverage gap 95%',
    group=MetricGroup.SUFFIX_LENGTH,
    unit=Unit.SCORE,
    direction=Direction.ZERO,
)
def suffix_length_coverage_gap_95(context: PreparedPrefix) -> float:
    """Return the 95% suffix-length central-interval coverage gap."""
    return coverage_gap(context.suffix_lengths, context.true_suffix_length, level=0.95)


@METRICS.register(
    'suffix_length_ae_point',
    label='Suffix length point absolute error',
    group=MetricGroup.SUFFIX_LENGTH,
    unit=Unit.EVENTS,
    direction=Direction.LOWER,
)
def suffix_length_ae_point(context: PreparedPrefix) -> float:
    """Return the point suffix-length absolute error."""
    generation = context.generation
    return float(abs(len(generation.point) - len(generation.truth)))
