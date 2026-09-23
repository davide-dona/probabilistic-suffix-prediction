from collections.abc import Sequence

from src.evaluation.metrics.definitions import Direction, MetricGroup, Unit
from src.evaluation.metrics.helpers.statistics import coverage_gap, crps
from src.evaluation.metrics.prepared import PreparedPrefix, aligned_inter_event_times
from src.evaluation.metrics.registry import METRICS

MINUTES_PER_DAY = 1440.0


@METRICS.register(
    'remaining_time_ae_point_days',
    label='Remaining-time point absolute error',
    group=MetricGroup.TIME,
    unit=Unit.DAYS,
    direction=Direction.LOWER,
)
def remaining_time_ae_point_days(context: PreparedPrefix) -> float:
    """Return the point remaining-time absolute error in days."""
    generation = context.generation
    return (
        abs(generation.point.remaining_time_minutes - generation.truth.remaining_time_minutes)
        / MINUTES_PER_DAY
    )


@METRICS.register(
    'inter_event_time_ae_point_days',
    label='Inter-event-time point absolute error',
    group=MetricGroup.TIME,
    unit=Unit.DAYS,
    direction=Direction.LOWER,
)
def inter_event_time_ae_point_days(context: PreparedPrefix) -> float:
    """Return the point inter-event-time absolute error in days."""
    generation = context.generation
    return (
        inter_event_time_ae_minutes(
            predicted=generation.point.inter_event_time_minutes,
            true=generation.truth.inter_event_time_minutes,
        )
        / MINUTES_PER_DAY
    )


@METRICS.register(
    'remaining_time_crps_days',
    label='Remaining-time CRPS',
    group=MetricGroup.TIME,
    unit=Unit.DAYS,
    direction=Direction.LOWER,
)
def remaining_time_crps_days(context: PreparedPrefix) -> float:
    """Return sampled remaining-time CRPS in days."""
    return crps(context.remaining_times, context.true_remaining_time) / MINUTES_PER_DAY


@METRICS.register(
    'inter_event_time_crps_days',
    label='Inter-event-time CRPS',
    group=MetricGroup.TIME,
    unit=Unit.DAYS,
    direction=Direction.LOWER,
)
def inter_event_time_crps_days(context: PreparedPrefix) -> float:
    """Return sampled inter-event-time CRPS in days."""
    return crps(context.inter_event_times, context.true_inter_event_times) / MINUTES_PER_DAY


@METRICS.register(
    'remaining_time_coverage_gap_50',
    label='Remaining time coverage gap 50%',
    group=MetricGroup.TIME,
    unit=Unit.SCORE,
    direction=Direction.ZERO,
)
def remaining_time_coverage_gap_50(context: PreparedPrefix) -> float:
    """Return the 50% remaining-time central-interval coverage gap."""
    return coverage_gap(context.remaining_times, context.true_remaining_time, level=0.50)


@METRICS.register(
    'remaining_time_coverage_gap_75',
    label='Remaining time coverage gap 75%',
    group=MetricGroup.TIME,
    unit=Unit.SCORE,
    direction=Direction.ZERO,
)
def remaining_time_coverage_gap_75(context: PreparedPrefix) -> float:
    """Return the 75% remaining-time central-interval coverage gap."""
    return coverage_gap(context.remaining_times, context.true_remaining_time, level=0.75)


@METRICS.register(
    'remaining_time_coverage_gap_95',
    label='Remaining time coverage gap 95%',
    group=MetricGroup.TIME,
    unit=Unit.SCORE,
    direction=Direction.ZERO,
)
def remaining_time_coverage_gap_95(context: PreparedPrefix) -> float:
    """Return the 95% remaining-time central-interval coverage gap."""
    return coverage_gap(context.remaining_times, context.true_remaining_time, level=0.95)


@METRICS.register(
    'inter_event_time_coverage_gap_50',
    label='Inter-event time coverage gap 50%',
    group=MetricGroup.TIME,
    unit=Unit.SCORE,
    direction=Direction.ZERO,
)
def inter_event_time_coverage_gap_50(context: PreparedPrefix) -> float:
    """Return the 50% inter-event-time central-interval coverage gap."""
    return coverage_gap(context.inter_event_times, context.true_inter_event_times, level=0.50)


@METRICS.register(
    'inter_event_time_coverage_gap_75',
    label='Inter-event time coverage gap 75%',
    group=MetricGroup.TIME,
    unit=Unit.SCORE,
    direction=Direction.ZERO,
)
def inter_event_time_coverage_gap_75(context: PreparedPrefix) -> float:
    """Return the 75% inter-event-time central-interval coverage gap."""
    return coverage_gap(context.inter_event_times, context.true_inter_event_times, level=0.75)


@METRICS.register(
    'inter_event_time_coverage_gap_95',
    label='Inter-event time coverage gap 95%',
    group=MetricGroup.TIME,
    unit=Unit.SCORE,
    direction=Direction.ZERO,
)
def inter_event_time_coverage_gap_95(context: PreparedPrefix) -> float:
    """Return the 95% inter-event-time central-interval coverage gap."""
    return coverage_gap(context.inter_event_times, context.true_inter_event_times, level=0.95)


def inter_event_time_ae_minutes(predicted: Sequence[float], true: Sequence[float]) -> float:
    """Return mean absolute error over the observed suffix positions."""
    if not true:
        return 0.0
    aligned = aligned_inter_event_times(predicted, length=len(true))
    errors = [abs(prediction - actual) for prediction, actual in zip(aligned, true, strict=True)]
    return sum(errors) / len(errors)
