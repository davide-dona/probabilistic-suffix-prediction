from dataclasses import dataclass
from typing import Self

from src.evaluation.activity_distances import SuffixMetric, sequence_similarity
from src.evaluation.activity_distances import energy_score as suffix_energy_score
from src.evaluation.scores.context import ScoringContext
from src.metrics import Direction, ScalarRecord, Unit, metric


@dataclass(frozen=True, slots=True)
class ActivityScores(ScalarRecord):
    """Scores for the generated activity suffixes."""

    dls_sample_mean: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)
    energy_score_dls: float = metric(unit=Unit.SCORE, direction=Direction.LOWER)
    energy_score_exact: float = metric(unit=Unit.SCORE, direction=Direction.LOWER)
    energy_score_bigram: float = metric(unit=Unit.SCORE, direction=Direction.LOWER)

    @classmethod
    def of(cls, context: ScoringContext) -> Self:
        """Score the sampled activity suffixes against the observed suffix."""
        samples, truth = context.generation.samples, context.generation.truth
        draws = len(samples)
        return cls(
            dls_sample_mean=(
                float(samples.counts @ context.similarities) / draws
                if context.similarities and draws
                else 0.0
            ),
            energy_score_dls=suffix_energy_score(
                samples.suffixes,
                truth.activities,
                weights=samples.counts,
                metric=SuffixMetric.DLD,
            ),
            energy_score_exact=suffix_energy_score(
                samples.suffixes,
                truth.activities,
                weights=samples.counts,
                metric=SuffixMetric.EXACT,
            ),
            energy_score_bigram=suffix_energy_score(
                samples.suffixes,
                truth.activities,
                weights=samples.counts,
                metric=SuffixMetric.BIGRAM,
            ),
        )


@dataclass(frozen=True, slots=True)
class ActivityDiagnostics(ScalarRecord):
    """Point-prediction activity score retained for run diagnostics."""

    dls_point: float

    @classmethod
    def of(cls, context: ScoringContext) -> Self:
        """Score the point activity suffix against the observed suffix."""
        generation = context.generation
        return cls(
            dls_point=sequence_similarity(generation.point.activities, generation.truth.activities)
        )
