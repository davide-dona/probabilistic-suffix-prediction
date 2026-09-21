from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain, islice, repeat
from typing import Self

import numpy as np

from src.inference.generation import Generation
from src.scalar_metrics import Direction, ScalarMetrics, Unit, mean, metric
from src.suffixes import SuffixMetric, sequence_similarity
from src.suffixes import energy_score as suffix_energy_score

MINUTES_PER_DAY = 1440.0
COVERAGE_LEVELS = (0.5, 0.75, 0.95)


@dataclass(frozen=True, slots=True)
class ScoringContext:
    """Decoded values shared by every accuracy family for one prefix."""

    generation: Generation
    similarities: tuple[float, ...]
    suffix_lengths: np.ndarray
    remaining_times: np.ndarray
    inter_event_times: np.ndarray
    true_suffix_length: np.ndarray
    true_remaining_time: np.ndarray
    true_inter_event_times: np.ndarray

    @classmethod
    def of(cls, generation: Generation) -> Self:
        """Prepare the shared draw and observation arrays for one prefix."""
        samples = generation.samples
        truth = generation.truth
        draws = len(samples)
        similarities = tuple(
            sequence_similarity(suffix, truth.activities) for suffix in samples.suffixes
        )
        suffix_lengths = np.array(
            [[float(len(events))] for events in samples.events], dtype=np.float64
        ).reshape(draws, 1)
        remaining_times = np.array(
            [[events.remaining_time_minutes] for events in samples.events], dtype=np.float64
        ).reshape(draws, 1)
        inter_event_times = np.array(
            [
                aligned_inter_event_times(events.inter_event_time_minutes, length=len(truth))
                for events in samples.events
            ],
            dtype=np.float64,
        ).reshape(draws, len(truth))
        return cls(
            generation=generation,
            similarities=similarities,
            suffix_lengths=suffix_lengths,
            remaining_times=remaining_times,
            inter_event_times=inter_event_times,
            true_suffix_length=np.array([float(len(truth))], dtype=np.float64),
            true_remaining_time=np.array(
                [truth.remaining_time_minutes], dtype=np.float64
            ),
            true_inter_event_times=np.array(
                truth.inter_event_time_minutes, dtype=np.float64
            ),
        )


@dataclass(frozen=True, slots=True)
class PointPredictionScores(ScalarMetrics):
    """Accuracy of the model's single prediction for one prefix."""

    dls_point: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)
    suffix_length_ae_point: float = metric(unit=Unit.EVENTS, direction=Direction.LOWER)
    remaining_time_ae_point_days: float = metric(unit=Unit.DAYS, direction=Direction.LOWER)
    inter_event_time_ae_point_days: float = metric(unit=Unit.DAYS, direction=Direction.LOWER)

    @classmethod
    def of(cls, context: ScoringContext) -> Self:
        point, truth = context.generation.point, context.generation.truth
        return cls(
            dls_point=sequence_similarity(point.activities, truth.activities),
            suffix_length_ae_point=float(abs(len(point) - len(truth))),
            remaining_time_ae_point_days=abs(
                point.remaining_time_minutes - truth.remaining_time_minutes
            )
            / MINUTES_PER_DAY,
            inter_event_time_ae_point_days=inter_event_time_ae_minutes(
                predicted=point.inter_event_time_minutes,
                true=truth.inter_event_time_minutes,
            )
            / MINUTES_PER_DAY,
        )


@dataclass(frozen=True, slots=True)
class SamplePredictionScores(ScalarMetrics):
    """Distributional accuracy of sampled suffixes for one prefix."""

    dls_sample_mean: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)
    energy_score_dls: float = metric(unit=Unit.SCORE, direction=Direction.LOWER)
    energy_score_exact: float = metric(unit=Unit.SCORE, direction=Direction.LOWER)
    energy_score_bigram: float = metric(unit=Unit.SCORE, direction=Direction.LOWER)
    suffix_length_crps: float = metric(unit=Unit.EVENTS, direction=Direction.LOWER)
    suffix_length_mae: float = metric(unit=Unit.EVENTS, direction=Direction.LOWER)
    remaining_time_crps_days: float = metric(unit=Unit.DAYS, direction=Direction.LOWER)
    inter_event_time_crps_days: float = metric(unit=Unit.DAYS, direction=Direction.LOWER)

    @classmethod
    def of(cls, context: ScoringContext) -> Self:
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
            suffix_length_crps=crps(context.suffix_lengths, context.true_suffix_length),
            suffix_length_mae=mae(context.suffix_lengths, context.true_suffix_length),
            remaining_time_crps_days=crps(
                context.remaining_times, context.true_remaining_time
            )
            / MINUTES_PER_DAY,
            inter_event_time_crps_days=crps(
                context.inter_event_times, context.true_inter_event_times
            )
            / MINUTES_PER_DAY,
        )


@dataclass(frozen=True, slots=True)
class CalibrationScores(ScalarMetrics):
    """Central-interval calibration gaps for one prefix."""

    suffix_length_coverage_gap_50: float = metric(unit=Unit.SCORE, direction=Direction.ZERO)
    suffix_length_coverage_gap_75: float = metric(unit=Unit.SCORE, direction=Direction.ZERO)
    suffix_length_coverage_gap_95: float = metric(unit=Unit.SCORE, direction=Direction.ZERO)
    remaining_time_coverage_gap_50: float = metric(unit=Unit.SCORE, direction=Direction.ZERO)
    remaining_time_coverage_gap_75: float = metric(unit=Unit.SCORE, direction=Direction.ZERO)
    remaining_time_coverage_gap_95: float = metric(unit=Unit.SCORE, direction=Direction.ZERO)
    inter_event_time_coverage_gap_50: float = metric(unit=Unit.SCORE, direction=Direction.ZERO)
    inter_event_time_coverage_gap_75: float = metric(unit=Unit.SCORE, direction=Direction.ZERO)
    inter_event_time_coverage_gap_95: float = metric(unit=Unit.SCORE, direction=Direction.ZERO)

    @classmethod
    def of(cls, context: ScoringContext) -> Self:
        suffix_length = coverage_gaps(context.suffix_lengths, context.true_suffix_length)
        remaining_time = coverage_gaps(context.remaining_times, context.true_remaining_time)
        inter_event_time = coverage_gaps(
            context.inter_event_times, context.true_inter_event_times
        )
        return cls(
            suffix_length_coverage_gap_50=suffix_length[0],
            suffix_length_coverage_gap_75=suffix_length[1],
            suffix_length_coverage_gap_95=suffix_length[2],
            remaining_time_coverage_gap_50=remaining_time[0],
            remaining_time_coverage_gap_75=remaining_time[1],
            remaining_time_coverage_gap_95=remaining_time[2],
            inter_event_time_coverage_gap_50=inter_event_time[0],
            inter_event_time_coverage_gap_75=inter_event_time[1],
            inter_event_time_coverage_gap_95=inter_event_time[2],
        )


def aligned_inter_event_times(predicted: Sequence[float], *, length: int) -> list[float]:
    """Pad or truncate inter-event times to the observed suffix length."""
    return list(islice(chain(predicted, repeat(0.0)), length))


def inter_event_time_ae_minutes(predicted: Sequence[float], true: Sequence[float]) -> float:
    """Return mean absolute error over the observed suffix positions."""
    if not true:
        return 0.0
    aligned = aligned_inter_event_times(predicted, length=len(true))
    return mean(
        [
            abs(prediction - actual)
            for prediction, actual in zip(aligned, true, strict=True)
        ]
    )


def crps(draws: np.ndarray, truth: np.ndarray) -> float:
    """Return mean CRPS over columns of a draw-by-column array."""
    count, columns = draws.shape
    if count == 0 or columns == 0:
        return 0.0
    accuracy = np.abs(draws - truth).mean(axis=0)
    if count == 1:
        return float(accuracy.mean())

    ordered = np.sort(draws, axis=0)
    ranks = np.arange(count, dtype=np.float64)[:, None]
    spread = ((2.0 * ranks - count + 1.0) * ordered).sum(axis=0)
    return float((accuracy - spread / (count * (count - 1.0))).mean())


def mae(draws: np.ndarray, truth: np.ndarray) -> float:
    """Return mean absolute error over draws and predicted quantities."""
    count, columns = draws.shape
    if count == 0 or columns == 0:
        return 0.0
    return float(np.abs(draws - truth).mean())


def coverage_gap(draws: np.ndarray, truth: np.ndarray, *, level: float) -> float:
    """Return empirical minus nominal central-interval coverage."""
    count, columns = draws.shape
    if columns == 0:
        return 0.0
    if count == 0:
        return -level
    low, high = np.quantile(
        draws, [(1.0 - level) / 2.0, (1.0 + level) / 2.0], axis=0
    )
    return float(((low <= truth) & (truth <= high)).mean()) - level


def coverage_gaps(draws: np.ndarray, truth: np.ndarray) -> tuple[float, ...]:
    """Return coverage gaps at the configured interval levels."""
    return tuple(coverage_gap(draws, truth, level=level) for level in COVERAGE_LEVELS)
