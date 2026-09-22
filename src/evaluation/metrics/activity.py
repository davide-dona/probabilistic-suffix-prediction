from src.evaluation.metrics.helpers.activity import SuffixMetric, energy_score, sequence_similarity
from src.evaluation.metrics.prepared import PreparedPrefix
from src.evaluation.metrics.registry import METRICS
from src.metrics import Direction, MetricGroup, Unit


@METRICS.register(
    'dls_sample_mean',
    label='DLS sample mean',
    group=MetricGroup.ACTIVITY,
    unit=Unit.SHARE,
    direction=Direction.HIGHER,
)
def dls_sample_mean(context: PreparedPrefix) -> float:
    """Return the draw-weighted DLS similarity of activity suffix samples."""
    samples = context.generation.samples
    draws = len(samples)
    return (
        float(samples.counts @ context.similarities) / draws
        if context.similarities and draws
        else 0.0
    )


@METRICS.register(
    'energy_score_dls',
    label='DLS energy score',
    group=MetricGroup.ACTIVITY,
    unit=Unit.SCORE,
    direction=Direction.LOWER,
)
def energy_score_dls(context: PreparedPrefix) -> float:
    """Return the sampled activity energy score on normalized DLS distance."""
    samples, truth = context.generation.samples, context.generation.truth
    return energy_score(
        samples.suffixes, truth.activities, weights=samples.counts, metric=SuffixMetric.DLD
    )


@METRICS.register(
    'energy_score_exact',
    label='Exact energy score',
    group=MetricGroup.ACTIVITY,
    unit=Unit.SCORE,
    direction=Direction.LOWER,
)
def energy_score_exact(context: PreparedPrefix) -> float:
    """Return the sampled activity energy score on exact-match distance."""
    samples, truth = context.generation.samples, context.generation.truth
    return energy_score(
        samples.suffixes, truth.activities, weights=samples.counts, metric=SuffixMetric.EXACT
    )


@METRICS.register(
    'energy_score_bigram',
    label='Bigram energy score',
    group=MetricGroup.ACTIVITY,
    unit=Unit.SCORE,
    direction=Direction.LOWER,
)
def energy_score_bigram(context: PreparedPrefix) -> float:
    """Return the sampled activity energy score on bigram distance."""
    samples, truth = context.generation.samples, context.generation.truth
    return energy_score(
        samples.suffixes, truth.activities, weights=samples.counts, metric=SuffixMetric.BIGRAM
    )


@METRICS.register(
    'dls_point',
    label='DLS point prediction',
    group=MetricGroup.ACTIVITY,
    unit=Unit.SHARE,
    direction=Direction.HIGHER,
)
def dls_point(context: PreparedPrefix) -> float:
    """Return the DLS similarity of the point activity prediction."""
    generation = context.generation
    return sequence_similarity(generation.point.activities, generation.truth.activities)
