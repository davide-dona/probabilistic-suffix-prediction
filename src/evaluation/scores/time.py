from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

from src.evaluation.scores.context import ScoringContext, aligned_inter_event_times
from src.evaluation.scores.suffix_length import coverage_gaps, crps
from src.metrics import ScalarRecord, mean

MINUTES_PER_DAY = 1440.0


@dataclass(frozen=True, slots=True)
class TimeDiagnostics(ScalarRecord):
    """Time-field scores retained for run diagnostics."""

    remaining_time_ae_point_days: float
    inter_event_time_ae_point_days: float
    remaining_time_crps_days: float
    inter_event_time_crps_days: float
    remaining_time_coverage_gap_50: float
    remaining_time_coverage_gap_75: float
    remaining_time_coverage_gap_95: float
    inter_event_time_coverage_gap_50: float
    inter_event_time_coverage_gap_75: float
    inter_event_time_coverage_gap_95: float

    @classmethod
    def of(cls, context: ScoringContext) -> Self:
        """Score generated timestamps against the observed suffix timestamps."""
        generation = context.generation
        point, truth = generation.point, generation.truth
        remaining_gaps = coverage_gaps(context.remaining_times, context.true_remaining_time)
        inter_event_gaps = coverage_gaps(context.inter_event_times, context.true_inter_event_times)
        return cls(
            remaining_time_ae_point_days=abs(
                point.remaining_time_minutes - truth.remaining_time_minutes
            )
            / MINUTES_PER_DAY,
            inter_event_time_ae_point_days=inter_event_time_ae_minutes(
                predicted=point.inter_event_time_minutes,
                true=truth.inter_event_time_minutes,
            )
            / MINUTES_PER_DAY,
            remaining_time_crps_days=crps(context.remaining_times, context.true_remaining_time)
            / MINUTES_PER_DAY,
            inter_event_time_crps_days=crps(
                context.inter_event_times, context.true_inter_event_times
            )
            / MINUTES_PER_DAY,
            remaining_time_coverage_gap_50=remaining_gaps[0],
            remaining_time_coverage_gap_75=remaining_gaps[1],
            remaining_time_coverage_gap_95=remaining_gaps[2],
            inter_event_time_coverage_gap_50=inter_event_gaps[0],
            inter_event_time_coverage_gap_75=inter_event_gaps[1],
            inter_event_time_coverage_gap_95=inter_event_gaps[2],
        )


def inter_event_time_ae_minutes(predicted: Sequence[float], true: Sequence[float]) -> float:
    """Return mean absolute error over the observed suffix positions."""
    if not true:
        return 0.0
    aligned = aligned_inter_event_times(predicted, length=len(true))
    return mean(
        [abs(prediction - actual) for prediction, actual in zip(aligned, true, strict=True)]
    )
