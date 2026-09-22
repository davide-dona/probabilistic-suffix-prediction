from src.evaluation.scores.context import ScoringContext
from src.evaluation.scores.registry import METRICS
from src.metrics import Direction, MetricGroup, Owner, Unit


def _sample_mean(context: ScoringContext, attribute: str) -> float:
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
def compute_conformance_sample_mean(context: ScoringContext) -> float:
    """Return the draw-weighted mean share of satisfied constraints."""
    return _sample_mean(context, 'share')


@METRICS.register(
    'conformance_point',
    label='Conformance point prediction',
    group=MetricGroup.CONFORMANCE,
    unit=Unit.SHARE,
    direction=Direction.HIGHER,
)
def compute_conformance_point(context: ScoringContext) -> float:
    """Return the satisfied-constraint share of the point prediction."""
    return context.point_conformance.share


@METRICS.register(
    'conformance_observed',
    label='Conformance observed',
    group=MetricGroup.CONFORMANCE,
    unit=Unit.SHARE,
    owner=Owner.LOG,
)
def compute_conformance_observed(context: ScoringContext) -> float:
    """Return the satisfied-constraint share of the observed continuation."""
    return context.observed_conformance.share


@METRICS.register(
    'full_conformance_sample_rate',
    label='Full conformance sample rate',
    group=MetricGroup.CONFORMANCE,
    unit=Unit.SHARE,
    direction=Direction.HIGHER,
)
def compute_full_conformance_sample_rate(context: ScoringContext) -> float:
    """Return the draw-weighted rate of fully conformant sampled suffixes."""
    return _sample_mean(context, 'full')


@METRICS.register(
    'full_conformance_observed',
    label='Full conformance observed',
    group=MetricGroup.CONFORMANCE,
    unit=Unit.SHARE,
    owner=Owner.LOG,
)
def compute_full_conformance_observed(context: ScoringContext) -> float:
    """Return whether the observed continuation fully conforms."""
    return context.observed_conformance.full
