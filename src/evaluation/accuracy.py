from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np

from src.inference.generation import Generation
from src.metrics import ScalarMetrics, mean
from src.suffixes import distances, sequence_similarity

MINUTES_PER_DAY = 1440.0


@dataclass(frozen=True, slots=True)
class AccuracyScores(ScalarMetrics):
    """The scores of one prefix's generated samples, or their mean over a set of prefixes."""

    # The Damerau-Levenshtein Similarity (DLS) between the samples and the ground truth [0, 1].
    dls_mean: float  # The mean similarity of a prefix's samples to the ground truth
    dls_point: (
        float  # z = mean(p(z | prefix), a single greedy answer, scored against the ground truth
    )
    # The closest of a prefix's samples to the ground truth.
    dls_best: float

    # The share of prefixes whose true suffix is exactly among their first k samples.
    hit_rate_at_1: float
    hit_rate_at_5: float
    hit_rate_at_10: float

    # The samples read as a predictive distribution, lower being better.
    # The score a checkpoint is selected on.
    energy_score: float

    # How far apart the samples of a prefix are, and how many of them are distinct sequences.
    # Both say how much of the prefix's uncertainty z carries, not how good the model is.
    sample_diversity: float
    unique_sample_rate: float

    # Absolute error (AE) between the predicted and true remaining cycle time, in days.
    remaining_time_ae_mean_days: float
    remaining_time_ae_point_days: float

    # Absolute error (AE) between the predicted and true suffix length, in events.
    length_ae_mean: float
    length_ae_point: float

    # Events left after the cut point: the scale every error above is read against. A property of
    # the prefixes scored rather than of the model, so it is flat across a training run.
    suffix_length: float


def diversity(samples: Sequence[Sequence[Hashable]]) -> float:
    """How far apart a set of sequences generated for one prefix are from each other, in `[0, 1]`.

    The mean distance over every pair, which is 0.0 when a prefix's samples are all the same
    sequence (or there are fewer than two to compare). Comparing samples of one prefix against
    each other is what measures the spread `p(z | prefix)` claims the prefix leaves open.

    Args:
        samples: The sequences generated for one prefix, one per draw of z.
    Returns:
        0.0 for identical (or singleton) sample sets, up to 1.0 for sequences sharing nothing.
    """
    if len(samples) < 2:
        return 0.0
    # Compute the full pairwise distance matrix
    pairs = distances(queries=samples, choices=samples, dtype=np.float64)
    # The matrix is symmetric with a diagonal of 0.0, so one triangle holds each pair once.
    rows, columns = np.triu_indices(n=len(samples), k=1)
    return mean(pairs[rows, columns].tolist())


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


def score_generation(generation: Generation) -> AccuracyScores:
    """
    Score the suffixes generated for one prefix against the ground truth they continue.

    Args:
        generation: The model's answer for one prefix, decoded into the log's own units.
    Returns:
        The prefix's scores. A prefix with no samples scores 0.0 on everything and 1.0 on
        `energy_score`, the worst it can be, rather than looking like a perfect prediction.
    """
    samples, point, truth = generation.samples, generation.point, generation.truth

    similarities = [sequence_similarity(sample.activities, truth.activities) for sample in samples]
    dls_mean = mean(similarities)
    sample_diversity = diversity([sample.activities for sample in samples])
    sample_activities = [tuple(sample.activities) for sample in samples]
    truth_activities = tuple(truth.activities)

    return AccuracyScores(
        dls_mean=dls_mean,
        dls_point=sequence_similarity(point.activities, truth.activities),
        dls_best=max(similarities, default=0.0),
        hit_rate_at_1=is_hit(samples=sample_activities, truth=truth_activities, k=1),
        hit_rate_at_5=is_hit(samples=sample_activities, truth=truth_activities, k=5),
        hit_rate_at_10=is_hit(samples=sample_activities, truth=truth_activities, k=10),
        energy_score=(
            energy_score(dls_mean=dls_mean, sample_diversity=sample_diversity) if samples else 1.0
        ),
        sample_diversity=sample_diversity,
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
        length_ae_mean=mean([float(abs(len(sample) - len(truth))) for sample in samples]),
        length_ae_point=float(abs(len(point) - len(truth))),
        suffix_length=float(len(truth)),
    )
