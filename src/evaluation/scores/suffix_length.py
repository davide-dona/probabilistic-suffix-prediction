import numpy as np

from src.evaluation.scores.context import ScoringContext
from src.evaluation.scores.registry import METRICS
from src.metrics import Direction, MetricGroup, Unit


@METRICS.register(
    'suffix_length_mae',
    label='Suffix length MAE',
    group=MetricGroup.SUFFIX_LENGTH,
    unit=Unit.EVENTS,
    direction=Direction.LOWER,
)
def compute_suffix_length_mae(context: ScoringContext) -> float:
    """Return the sampled suffix-length mean absolute error."""
    return mae(context.suffix_lengths, context.true_suffix_length)


@METRICS.register(
    'suffix_length_crps',
    label='Suffix length CRPS',
    group=MetricGroup.SUFFIX_LENGTH,
    unit=Unit.EVENTS,
    direction=Direction.LOWER,
)
def compute_suffix_length_crps(context: ScoringContext) -> float:
    """Return the sampled suffix-length CRPS."""
    return crps(context.suffix_lengths, context.true_suffix_length)


@METRICS.register(
    'suffix_length_coverage_gap_50',
    label='Suffix length coverage gap 50%',
    group=MetricGroup.SUFFIX_LENGTH,
    unit=Unit.SCORE,
    direction=Direction.ZERO,
)
def compute_suffix_length_coverage_gap_50(context: ScoringContext) -> float:
    """Return the 50% suffix-length central-interval coverage gap."""
    return coverage_gap(context.suffix_lengths, context.true_suffix_length, level=0.50)


@METRICS.register(
    'suffix_length_coverage_gap_75',
    label='Suffix length coverage gap 75%',
    group=MetricGroup.SUFFIX_LENGTH,
    unit=Unit.SCORE,
    direction=Direction.ZERO,
)
def compute_suffix_length_coverage_gap_75(context: ScoringContext) -> float:
    """Return the 75% suffix-length central-interval coverage gap."""
    return coverage_gap(context.suffix_lengths, context.true_suffix_length, level=0.75)


@METRICS.register(
    'suffix_length_coverage_gap_95',
    label='Suffix length coverage gap 95%',
    group=MetricGroup.SUFFIX_LENGTH,
    unit=Unit.SCORE,
    direction=Direction.ZERO,
)
def compute_suffix_length_coverage_gap_95(context: ScoringContext) -> float:
    """Return the 95% suffix-length central-interval coverage gap."""
    return coverage_gap(context.suffix_lengths, context.true_suffix_length, level=0.95)


@METRICS.register(
    'suffix_length_ae_point',
    label='Suffix length point absolute error',
    group=MetricGroup.SUFFIX_LENGTH,
    unit=Unit.EVENTS,
    direction=Direction.LOWER,
)
def compute_suffix_length_ae_point(context: ScoringContext) -> float:
    """Return the point suffix-length absolute error."""
    generation = context.generation
    return float(abs(len(generation.point) - len(generation.truth)))


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


def coverage_gap(draws: np.ndarray, truth: np.ndarray, *, level: float) -> float:
    """Return empirical minus nominal central-interval coverage."""
    count, columns = draws.shape
    if columns == 0:
        return 0.0
    if count == 0:
        return -level
    low, high = np.quantile(draws, [(1.0 - level) / 2.0, (1.0 + level) / 2.0], axis=0)
    return float(((low <= truth) & (truth <= high)).mean()) - level
