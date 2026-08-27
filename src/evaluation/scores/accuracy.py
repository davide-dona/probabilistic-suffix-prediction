from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain, islice, repeat
from typing import Self

from src.inference.generation import Generation
from src.scalar_metrics import ScalarMetrics, Unit, mean, metric
from src.suffixes import sequence_similarity, spread

MINUTES_PER_DAY = 1440.0


@dataclass(frozen=True, slots=True)
class AccuracyScores(ScalarMetrics):
    """How close a prefix's generated suffixes are to the one that actually followed it.

    The scores of one prefix's generated samples, or their mean over a set of prefixes.
    """

    # The Damerau-Levenshtein Similarity (DLS) between the samples and the ground truth.
    # The mean similarity of a prefix's samples to the ground truth
    dls_mean: float = metric(unit=Unit.SHARE, higher_is_better=True)
    # z = mean(p(z | prefix), a single greedy answer, scored against the ground truth
    dls_point: float = metric(unit=Unit.SHARE, higher_is_better=True)
    # The closest of a prefix's samples to the ground truth.
    dls_best: float = metric(unit=Unit.SHARE, higher_is_better=True)

    # The share of prefixes whose true suffix is exactly among their first k samples.
    hit_rate_at_1: float = metric(unit=Unit.SHARE, higher_is_better=True)
    hit_rate_at_5: float = metric(unit=Unit.SHARE, higher_is_better=True)
    hit_rate_at_10: float = metric(unit=Unit.SHARE, higher_is_better=True)

    # The samples read as a predictive distribution, lower being better.
    # The score a checkpoint is selected on. Runs from -0.5 to 1.0, so it is no share.
    energy_score: float = metric(unit=Unit.SCORE, higher_is_better=False)

    # How far apart the samples of a prefix are, and how many of them are distinct sequences.
    # Both say how much of the prefix's uncertainty z carries, not how good the model is.
    sample_diversity: float = metric(unit=Unit.SHARE)
    unique_sample_rate: float = metric(unit=Unit.SHARE)

    # Absolute error (AE) between the predicted and true remaining cycle time, in days.
    remaining_time_ae_mean_days: float = metric(unit=Unit.DAYS, higher_is_better=False)
    remaining_time_ae_point_days: float = metric(unit=Unit.DAYS, higher_is_better=False)

    # Absolute error (AE) between the predicted and true minutes until each generated event,
    # averaged over the positions the true suffix covers, in days.
    time_to_next_ae_mean_days: float = metric(unit=Unit.DAYS, higher_is_better=False)
    time_to_next_ae_point_days: float = metric(unit=Unit.DAYS, higher_is_better=False)

    # Absolute error (AE) between the predicted and true suffix length, in events.
    length_ae_mean: float = metric(unit=Unit.EVENTS, higher_is_better=False)
    length_ae_point: float = metric(unit=Unit.EVENTS, higher_is_better=False)

    # Events left after the cut point: the scale every error above is read against. A property of
    # the prefixes scored rather than of the model, so it is flat across a training run.
    suffix_length: float = metric(unit=Unit.EVENTS)

    @classmethod
    def of(cls, generation: Generation) -> Self:
        """Score the suffixes generated for one prefix against the ground truth they continue.

        Args:
            generation: The model's answer for one prefix, decoded into the log's own units.
        Returns:
            The prefix's scores. A prefix with no samples scores 0.0 on everything and 1.0 on
            `energy_score`, the worst it can be, rather than looking like a perfect prediction.
        """
        samples, point, truth = generation.samples, generation.point, generation.truth

        similarities = [
            sequence_similarity(sample.activities, truth.activities) for sample in samples
        ]
        dls_mean = mean(similarities)
        # Comparing a prefix's samples against each other is what measures the spread
        # `p(z | prefix)` claims the prefix leaves open.
        sample_spread = spread([sample.activities for sample in samples])
        sample_activities = [tuple(sample.activities) for sample in samples]
        truth_activities = tuple(truth.activities)

        return cls(
            dls_mean=dls_mean,
            dls_point=sequence_similarity(point.activities, truth.activities),
            dls_best=max(similarities, default=0.0),
            hit_rate_at_1=is_hit(samples=sample_activities, truth=truth_activities, k=1),
            hit_rate_at_5=is_hit(samples=sample_activities, truth=truth_activities, k=5),
            hit_rate_at_10=is_hit(samples=sample_activities, truth=truth_activities, k=10),
            energy_score=(
                energy_score(dls_mean=dls_mean, sample_diversity=sample_spread) if samples else 1.0
            ),
            sample_diversity=sample_spread,
            unique_sample_rate=len(set(sample_activities)) / len(samples) if samples else 0.0,
            remaining_time_ae_mean_days=mean(
                [
                    abs(sample.remaining_time_minutes - truth.remaining_time_minutes)
                    for sample in samples
                ]
            )
            / MINUTES_PER_DAY,
            remaining_time_ae_point_days=abs(
                point.remaining_time_minutes - truth.remaining_time_minutes
            )
            / MINUTES_PER_DAY,
            time_to_next_ae_mean_days=mean(
                [
                    time_to_next_ae_minutes(
                        predicted=sample.time_to_next_minutes,
                        true=truth.time_to_next_minutes,
                    )
                    for sample in samples
                ]
            )
            / MINUTES_PER_DAY,
            time_to_next_ae_point_days=time_to_next_ae_minutes(
                predicted=point.time_to_next_minutes,
                true=truth.time_to_next_minutes,
            )
            / MINUTES_PER_DAY,
            length_ae_mean=mean([float(abs(len(sample) - len(truth))) for sample in samples]),
            length_ae_point=float(abs(len(point) - len(truth))),
            suffix_length=float(len(truth)),
        )


def time_to_next_ae_minutes(predicted: Sequence[float], true: Sequence[float]) -> float:
    """Mean absolute error between a run's waits until each event and the true ones, in minutes.

    The true suffix sets the range the error is read over: a run that ended early counts as 0
    minutes at every position it did not write, and anything it wrote past the true end is dropped.

    Args:
        predicted: The generated minutes until each event of one run.
        true: The minutes until each event of the true suffix.
    Returns:
        The mean absolute error over the `len(true)` positions, or 0.0 for an empty true suffix.
    """
    if not true:
        return 0.0
    padded = islice(chain(predicted, repeat(0.0)), len(true))
    return mean([abs(prediction - actual) for prediction, actual in zip(padded, true, strict=True)])


def energy_score(dls_mean: float, sample_diversity: float) -> float:
    """A score balancing how close a prefix's samples are to the truth
    and how far apart they are from each other."""
    return (1.0 - dls_mean) - 0.5 * sample_diversity


def is_hit(samples: Sequence[tuple[str, ...]], truth: tuple[str, ...], *, k: int) -> float:
    """
    Whether the true sequence is among the first `k` samples, exactly.

    Args:
        samples: The activity sequences generated for one prefix, in the order they were drawn.
            A draw is independent of the ones before it, so the first `k` of them are as good a
            sample of `k` as any other choice.
        truth: The ground-truth activity sequence.
        k: How many samples to look at.
    Returns:
        1.0 if one of the first `k` samples is the true sequence, 0.0 otherwise.
    """
    return float(truth in samples[:k])
