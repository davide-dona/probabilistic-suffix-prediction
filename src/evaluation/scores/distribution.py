from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np
import ot
from scipy.stats import wasserstein_distance

from src.evaluation.scores.accuracy import MINUTES_PER_DAY
from src.inference.generation import Generation
from src.logs.continuations import ContinuationIndex, References
from src.scalar_metrics import Direction, Owner, ScalarMetrics, Unit, mean, metric
from src.suffixes import distances, spread


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

    # The comparison on the three marginals a suffix carries beyond its activities. The waits
    # are grouped by the activity they precede rather than by the position they fell at: a process
    # constrains how long an activity takes, where a position only means something once the
    # control flow is already right, which the three columns above are what answer.
    length_wasserstein: float = metric(unit=Unit.EVENTS, direction=Direction.LOWER)
    remaining_time_wasserstein_days: float = metric(unit=Unit.DAYS, direction=Direction.LOWER)
    activity_time_wasserstein_days: float = metric(unit=Unit.DAYS, direction=Direction.LOWER)

    # How far apart the samples of a prefix are, and how many of them are distinct sequences.
    # Neither has a best value of its own: the spread a model should have is the spread the log's
    # own continuations have, which `reference_diversity` below carries.
    sample_diversity: float = metric(unit=Unit.SHARE)
    unique_sample_rate: float = metric(unit=Unit.SHARE)

    # How far apart two of the prefix's observed continuations are, which is the scale
    # `sample_diversity` is read against: a model matching the log's multimodality spreads its
    # draws about this far apart, no further and no less. A property of the log rather than of the
    # model, so it is the same for every model of a log and is never compared between them.
    reference_diversity: float = metric(unit=Unit.SHARE, owner=Owner.LOG)

    # How many distinct continuations the log took after this prefix. A property of the log rather
    # than of the model, so it is the same for every model of a log.
    reference_size: float = metric(unit=Unit.COUNT, owner=Owner.LOG)

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

        # A wait belongs to the activity it precedes, so the suffixes encoded above are read a
        # character at a time against the waits the same draw was written with. `_decode` cuts a
        # run's activities and its waits to one length, so the two always pair up.
        drawn: dict[str, list[float]] = {}
        for suffix, sample in zip(suffixes, generation.samples, strict=True):
            for activity, wait in zip(suffix, sample.time_to_next_minutes, strict=True):
                drawn.setdefault(activity, []).append(wait)

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
            activity_time_wasserstein_days=activity_time_wasserstein_minutes(
                generated=drawn, observed=references.waits
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
            reference_size=float(len(references.suffixes)),
        )


def emsc(suffixes: tuple[str, ...], references: References) -> float:
    """Earth Movers' Stochastic Conformance between generated and observed continuations.

    The two sets of suffixes are read as stochastic languages, the generated one uniform over the
    draws and the observed one weighted by how often each continuation occurred, and compared by
    the optimal transport cost of turning the first into the second under the normalized
    Damerau-Levenshtein distance. Reported as a similarity so it reads the way every other share
    here does.

    Solved over every continuation the prefix was observed to take. The short prefixes of a large
    log run to a few thousand of them, which the exact solver handles in tens of milliseconds, and
    dropping the light tail is not free: those prefixes are the ones a log revisits most, so an
    approximation there would move the mean of a tenth of the split.

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

    # `[len(suffixes), len(references.suffixes)]`, one row per draw and one column per distinct
    # continuation.
    cost = distances(queries=suffixes, choices=references.suffixes, dtype=np.float64)
    draws = np.full(len(suffixes), 1.0 / len(suffixes))
    weights = references.weights / references.occurrences
    return 1.0 - float(ot.emd2(a=draws, b=weights, M=cost))


def activity_time_wasserstein_minutes(
    generated: Mapping[str, Sequence[float]], observed: Mapping[str, np.ndarray]
) -> float:
    """How far the waits a model puts before each activity are from the ones the log put there.

    One 1-Wasserstein distance per activity, between the waits pooled over every draw and the ones
    pooled over every occurrence of the prefix, averaged with each activity weighed by how often
    the log ran it. Grouping by the activity is what makes each of them a comparison of two
    conditional distributions, so a draw running longer or shorter than an occurrence normalizes
    away rather than leaking into the timing. Read by position instead, one inserted event shifts
    every wait after it and the number reports as a timing error what the activities were wrong
    about.

    An activity only one side ran is skipped. Writing an activity the log never took after this
    prefix, or never writing one it did, is a control-flow error, and `emsc`,
    `continuation_precision` and `continuation_recall` are what charge for it; counting it here
    would restate it as a timing error, which is the same mistake reading the waits by position
    makes.

    Both sides are small, so this is biased upward the way `length_wasserstein` and
    `remaining_time_wasserstein_days` are. The bias follows the draw count and the prefix's
    occurrence count, both of which every model of a log shares, so it is a number to compare
    models on rather than a distance to quote on its own.

    Args:
        generated: The waits of every draw, pooled under the activity each of them precedes.
        observed: The same over every continuation the prefix was observed to take.
    Returns:
        The weighted mean distance, in minutes. Where the two share no activity the pools are
        compared unconditioned instead, since 0.0 would read as a perfect match on a score that
        has no worst value of its own.
    """
    scored = [
        (float(len(waits)), wasserstein_distance(u_values=generated[activity], v_values=waits))
        for activity, waits in observed.items()
        if activity in generated
    ]
    if scored:
        return sum(weight * distance for weight, distance in scored) / sum(
            weight for weight, _ in scored
        )
    return wasserstein_distance(
        u_values=[wait for waits in generated.values() for wait in waits] or [0.0],
        v_values=[wait for waits in observed.values() for wait in waits],
    )
