from dataclasses import dataclass
from typing import Self

import numpy as np
import ot
from scipy.stats import wasserstein_distance

from src.evaluation.scores.accuracy import MINUTES_PER_DAY
from src.inference.generation import Generation
from src.logs.continuations import ContinuationIndex, References
from src.scalar_metrics import Direction, ScalarMetrics, Unit, mean, metric
from src.suffixes import distances, spread

# How many distinct continuations one prefix's transport problem is solved over. A prefix of length
# one on a large log is followed by tens of thousands of them, and the exact solver is superlinear
# in that count, so beyond this the heaviest are kept and their mass renormalized. Set well above
# the mean reference size of every log here, so it is a guard against the tail rather than an
# approximation the usual case runs through.
_MAX_REFERENCES = 2048


@dataclass(frozen=True, slots=True)
class DistributionScores(ScalarMetrics):
    """How close the distribution of suffixes generated for a prefix is to the distribution of
    the ones the log was observed to take after it, and how far apart each of the two spreads its
    own continuations."""

    # Earth Movers' Stochastic Conformance between the two distributions, as a similarity: the
    # optimal transport cost of turning the generated suffixes into the observed ones, subtracted
    # from 1, so 1.0 means the model reproduces the prefix's continuations exactly.
    emsc: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)

    # The precision and recall of the samples read against the observed continuations. Recall is
    # weighted by how often each continuation was observed, so it is the share of what actually
    # happens that the model reproduces; precision counts a draw as a hit whether the log took that
    # continuation once or five hundred times.
    continuation_recall: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)
    continuation_precision: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)

    # The comparison on the two marginals a suffix carries beyond its activities.
    length_wasserstein: float = metric(unit=Unit.EVENTS, direction=Direction.LOWER)
    remaining_time_wasserstein_days: float = metric(unit=Unit.DAYS, direction=Direction.LOWER)

    # How far apart the samples of a prefix are, and how many of them are distinct sequences.
    # Neither has a best value of its own: the spread a model should have is the spread the log's
    # own continuations have, which `reference_diversity` below carries.
    sample_diversity: float = metric(unit=Unit.SHARE)
    unique_sample_rate: float = metric(unit=Unit.SHARE)

    # How far apart two of the prefix's observed continuations are, which is the scale
    # `sample_diversity` is read against: a model matching the log's multimodality spreads its
    # draws about this far apart, no further and no less. A property of the log rather than of the
    # model, so it is the same for every model of a log and is never compared between them.
    reference_diversity: float = metric(unit=Unit.SHARE)

    # `sample_diversity` read against it. Positive means the model spreads its draws wider than the
    # process does, which is as wrong as collapsing them onto one mode, so the target is 0 and the
    # best of a row is the one nearest it.
    diversity_gap: float = metric(unit=Unit.SCORE, direction=Direction.ZERO)

    # How many distinct continuations the log took after this prefix. A property of the log rather
    # than of the model, so it is the same for every model of a log.
    reference_size: float = metric(unit=Unit.COUNT)

    @classmethod
    def of(cls, generation: Generation, *, index: ContinuationIndex) -> Self:
        """Score the suffixes generated for one prefix against every suffix observed after it.

        Args:
            generation: The model's answer for one prefix, decoded into the log's own units.
            index: The log's observed continuations, held by the process doing the scoring.
        Returns:
            The prefix's scores.
        """
        references = index.references(generation.prefix_activities)
        suffixes = tuple(index.encode(sample.activities) for sample in generation.samples)

        # Comparing a prefix's samples against each other is what measures the spread
        # `p(z | prefix)` claims the prefix leaves open.
        sample_spread = spread([sample.activities for sample in generation.samples])

        observed, generated = set(references.suffixes), set(suffixes)
        covered = sum(
            weight
            for suffix, weight in zip(references.suffixes, references.weights, strict=True)
            if suffix in generated
        )

        lengths = [float(len(sample)) for sample in generation.samples] or [0.0]
        remaining = [sample.remaining_time_minutes for sample in generation.samples] or [0.0]

        return cls(
            emsc=emsc(suffixes=suffixes, references=references),
            continuation_recall=covered / references.occurrences,
            continuation_precision=mean([float(suffix in observed) for suffix in suffixes]),
            # A continuation's length is fixed by its activities, so the distinct suffixes weighed
            # by their counts are the same distribution as one length per occurrence.
            length_wasserstein=wasserstein_distance(
                u_values=lengths,
                v_values=[float(len(suffix)) for suffix in references.suffixes],
                v_weights=references.weights,
            ),
            remaining_time_wasserstein_days=wasserstein_distance(
                u_values=remaining,
                v_values=references.remaining_times,
            )
            / MINUTES_PER_DAY,
            sample_diversity=sample_spread,
            unique_sample_rate=(
                len({tuple(sample.activities) for sample in generation.samples})
                / len(generation.samples)
                if generation.samples
                else 0.0
            ),
            reference_diversity=references.dispersion,
            # A mean of differences is the difference of the means, so a gap column and the two
            # level columns it is read from stay consistent at every granularity.
            diversity_gap=sample_spread - references.dispersion,
            reference_size=float(len(references.suffixes)),
        )


def emsc(suffixes: tuple[str, ...], references: References) -> float:
    """Earth Movers' Stochastic Conformance between generated and observed continuations.

    The two sets of suffixes are read as stochastic languages, the generated one uniform over the
    draws and the observed one weighted by how often each continuation occurred, and compared by
    the optimal transport cost of turning the first into the second under the normalized
    Damerau-Levenshtein distance. Reported as a similarity so it reads the way every other share
    here does.

    Args:
        suffixes: The generated suffixes, encoded onto the index's scale, one per draw.
        references: The prefix's observed continuations, their weights and their remaining times.
    Returns:
        `1 - cost` in `[0, 1]`, 1.0 where the model reproduces the prefix's continuations exactly
        and 0.0 where no draw shares anything with any of them. 0.0 for a prefix with no draws,
        the worst it can be, rather than looking like a perfect match.
    """
    if not suffixes:
        return 0.0

    choices, weights = _heaviest(references)
    # `[len(suffixes), len(choices)]`, one row per draw and one column per distinct continuation.
    cost = distances(queries=suffixes, choices=choices, dtype=np.float64)
    draws = np.full(len(suffixes), 1.0 / len(suffixes))
    return 1.0 - float(ot.emd2(a=draws, b=weights / weights.sum(), M=cost))


def _heaviest(references: References) -> tuple[tuple[str, ...], np.ndarray]:
    """The continuations one transport problem is solved over, and how often each was observed.

    Args:
        references: The prefix's observed continuations.
    Returns:
        Every continuation and its weight where the prefix has no more than `_MAX_REFERENCES` of
        them, and otherwise the most frequent `_MAX_REFERENCES`, which carry the bulk of the mass.
    """
    if len(references.suffixes) <= _MAX_REFERENCES:
        return references.suffixes, references.weights

    kept = np.argsort(references.weights)[-_MAX_REFERENCES:]
    return tuple(references.suffixes[row] for row in kept), references.weights[kept]
