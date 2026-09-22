from src.evaluation.activity_distances import SuffixMetric, sequence_similarity
from src.evaluation.activity_distances import energy_score as suffix_energy_score
from src.evaluation.scores.context import ScoringContext
from src.evaluation.scores.registry import METRICS
from src.metrics import Direction, MetricGroup, Unit


@METRICS.register(
    'dls_sample_mean',
    label='DLS sample mean',
    group=MetricGroup.ACTIVITY,
    unit=Unit.SHARE,
    direction=Direction.HIGHER,
)
def compute_dls_sample_mean(context: ScoringContext) -> float:
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
def compute_energy_score_dls(context: ScoringContext) -> float:
    """Return the sampled activity energy score on normalized DLS distance."""
    samples, truth = context.generation.samples, context.generation.truth
    return suffix_energy_score(
        samples.suffixes, truth.activities, weights=samples.counts, metric=SuffixMetric.DLD
    )


@METRICS.register(
    'energy_score_exact',
    label='Exact energy score',
    group=MetricGroup.ACTIVITY,
    unit=Unit.SCORE,
    direction=Direction.LOWER,
)
def compute_energy_score_exact(context: ScoringContext) -> float:
    """Return the sampled activity energy score on exact-match distance."""
    samples, truth = context.generation.samples, context.generation.truth
    return suffix_energy_score(
        samples.suffixes, truth.activities, weights=samples.counts, metric=SuffixMetric.EXACT
    )


@METRICS.register(
    'energy_score_bigram',
    label='Bigram energy score',
    group=MetricGroup.ACTIVITY,
    unit=Unit.SCORE,
    direction=Direction.LOWER,
)
def compute_energy_score_bigram(context: ScoringContext) -> float:
    """Return the sampled activity energy score on bigram distance."""
    samples, truth = context.generation.samples, context.generation.truth
    return suffix_energy_score(
        samples.suffixes, truth.activities, weights=samples.counts, metric=SuffixMetric.BIGRAM
    )


@METRICS.register(
    'dls_point',
    label='DLS point prediction',
    group=MetricGroup.ACTIVITY,
    unit=Unit.SHARE,
    direction=Direction.HIGHER,
)
def compute_dls_point(context: ScoringContext) -> float:
    """Return the DLS similarity of the point activity prediction."""
    generation = context.generation
    return sequence_similarity(generation.point.activities, generation.truth.activities)
