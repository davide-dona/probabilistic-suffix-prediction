from src.evaluation.metrics.definitions import Direction, MetricGroup, Owner, Unit
from src.evaluation.metrics.prepared import PreparedPrefix
from src.evaluation.metrics.registry import METRICS


def _sample_mean(context: PreparedPrefix, attribute: str) -> float:
    """Return one conformance property averaged across all sampled draws."""
    checks = context.sample_conformance
    samples = context.generation.samples
    draws = len(samples)
    values = [getattr(check, attribute) for check in checks]
    return float(samples.counts @ values) / draws if values and draws else 0.0


@METRICS.register(
    'conformance_sample_mean',
    label='Conformance sample mean',
    group=MetricGroup.CONFORMANCE,
    unit=Unit.SHARE,
    direction=Direction.HIGHER,
)
def conformance_sample_mean(context: PreparedPrefix) -> float:
    """Return the draw-weighted mean share of satisfied constraints."""
    return _sample_mean(context, 'share')


@METRICS.register(
    'conformance_point',
    label='Conformance point prediction',
    group=MetricGroup.CONFORMANCE,
    unit=Unit.SHARE,
    direction=Direction.HIGHER,
)
def conformance_point(context: PreparedPrefix) -> float:
    """Return the satisfied-constraint share of the point prediction."""
    return context.point_conformance.share


@METRICS.register(
    'conformance_observed',
    label='Conformance observed',
    group=MetricGroup.CONFORMANCE,
    unit=Unit.SHARE,
    owner=Owner.LOG,
)
def conformance_observed(context: PreparedPrefix) -> float:
    """Return the satisfied-constraint share of the observed continuation."""
    return context.observed_conformance.share


@METRICS.register(
    'full_conformance_sample_rate',
    label='Full conformance sample rate',
    group=MetricGroup.CONFORMANCE,
    unit=Unit.SHARE,
    direction=Direction.HIGHER,
)
def full_conformance_sample_rate(context: PreparedPrefix) -> float:
    """Return the draw-weighted rate of fully conformant sampled suffixes."""
    return _sample_mean(context, 'full')


@METRICS.register(
    'full_conformance_observed',
    label='Full conformance observed',
    group=MetricGroup.CONFORMANCE,
    unit=Unit.SHARE,
    owner=Owner.LOG,
)
def full_conformance_observed(context: PreparedPrefix) -> float:
    """Return whether the observed continuation fully conforms."""
    return context.observed_conformance.full
