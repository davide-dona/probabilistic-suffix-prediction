from dataclasses import dataclass
from typing import Self

import numpy as np

from src.evaluation.scores.context import ScoringContext
from src.metrics import Direction, ScalarRecord, Unit, metric

COVERAGE_LEVELS = (0.50, 0.75, 0.95)


@dataclass(frozen=True, slots=True)
class SuffixLengthScores(ScalarRecord):
    """Scores for the sampled suffix-length distribution."""

    suffix_length_mae: float = metric(unit=Unit.EVENTS, direction=Direction.LOWER)
    suffix_length_crps: float = metric(unit=Unit.EVENTS, direction=Direction.LOWER)
    suffix_length_coverage_gap_50: float = metric(unit=Unit.SCORE, direction=Direction.ZERO)
    suffix_length_coverage_gap_75: float = metric(unit=Unit.SCORE, direction=Direction.ZERO)
    suffix_length_coverage_gap_95: float = metric(unit=Unit.SCORE, direction=Direction.ZERO)

    @classmethod
    def of(cls, context: ScoringContext) -> Self:
        """Score accuracy and calibration of generated suffix lengths."""
        gaps = coverage_gaps(context.suffix_lengths, context.true_suffix_length)
        return cls(
            suffix_length_mae=mae(context.suffix_lengths, context.true_suffix_length),
            suffix_length_crps=crps(context.suffix_lengths, context.true_suffix_length),
            suffix_length_coverage_gap_50=gaps[0],
            suffix_length_coverage_gap_75=gaps[1],
            suffix_length_coverage_gap_95=gaps[2],
        )


@dataclass(frozen=True, slots=True)
class SuffixLengthDiagnostics(ScalarRecord):
    """Point-prediction suffix-length error retained for run diagnostics."""

    suffix_length_ae_point: float

    @classmethod
    def of(cls, context: ScoringContext) -> Self:
        """Score the point suffix length against the observed suffix length."""
        generation = context.generation
        return cls(suffix_length_ae_point=float(abs(len(generation.point) - len(generation.truth))))


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
    return float(np.abs(draws - truth).mean()) if count and columns else 0.0


def coverage_gaps(draws: np.ndarray, truth: np.ndarray) -> tuple[float, ...]:
    """Return central-interval coverage gaps at the configured levels."""
    return tuple(coverage_gap(draws, truth, level=level) for level in COVERAGE_LEVELS)


def coverage_gap(draws: np.ndarray, truth: np.ndarray, *, level: float) -> float:
    """Return empirical minus nominal central-interval coverage."""
    count, columns = draws.shape
    if columns == 0:
        return 0.0
    if count == 0:
        return -level
    low, high = np.quantile(draws, [(1.0 - level) / 2.0, (1.0 + level) / 2.0], axis=0)
    return float(((low <= truth) & (truth <= high)).mean()) - level
