from dataclasses import dataclass
from typing import Self

import numpy as np
from scipy.stats import wasserstein_distance

from src.evaluation.scores.accuracy import MINUTES_PER_DAY
from src.inference.generation import Generation
from src.logs.continuations import ContinuationIndex
from src.scalar_metrics import ScalarMetrics, Unit, mean, metric
from src.suffixes import distances, spread


@dataclass(frozen=True, slots=True)
class DistributionScores(ScalarMetrics):
    """How close the distribution of suffixes generated for a prefix is to the distribution of
    the ones the log was observed to take after it."""

    # How much of the observed distribution the model's samples cover
    coverage: float = metric(unit=Unit.SHARE, higher_is_better=True)
    # How much of the model's samples are observed in the log
    precision: float = metric(unit=Unit.SHARE, higher_is_better=True)

    # How far the two distributions are from each other, on the scale `AccuracyScores.energy_score`
    # reads on: the two are the same number where the log took one continuation after the prefix.
    reference_energy_score: float = metric(unit=Unit.SCORE, higher_is_better=False)

    # The same comparison on the two marginals a suffix carries beyond its activities.
    length_wasserstein: float = metric(unit=Unit.EVENTS, higher_is_better=False)
    remaining_time_wasserstein_days: float = metric(unit=Unit.DAYS, higher_is_better=False)

    # How many distinct continuations the log took after this prefix. A property of the log rather
    # than of the model, so it is the same in every model's column.
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
        occurrences = references.occurrences

        observed, generated = set(references.suffixes), set(suffixes)
        covered = sum(
            weight
            for suffix, weight in zip(references.suffixes, references.weights, strict=True)
            if suffix in generated
        )

        # `[len(suffixes), len(references.suffixes)]`, one row per draw and one column per
        # distinct continuation.
        pairs = distances(queries=suffixes, choices=references.suffixes, dtype=np.float64)
        cross = float((pairs @ references.weights).sum()) / (len(suffixes) * occurrences)

        lengths = [float(len(sample)) for sample in generation.samples] or [0.0]
        remaining = [sample.remaining_time_minutes for sample in generation.samples] or [0.0]

        return cls(
            coverage=covered / occurrences,
            precision=mean([float(suffix in observed) for suffix in suffixes]),
            # Both within-set terms read every ordered pair of two distinct draws, so neither is a
            # function of how many were taken.
            reference_energy_score=cross - 0.5 * spread(suffixes) - 0.5 * references.dispersion,
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
            reference_size=float(len(references.suffixes)),
        )
