from collections.abc import Mapping

from src.evaluation.metrics import METRICS
from src.evaluation.metrics.metadata import Direction, Metric

SELECTION_METRIC: Metric = METRICS['energy_score_dls']

if SELECTION_METRIC.direction is not Direction.LOWER:
    raise ValueError(f'{SELECTION_METRIC.key} must be minimized for model selection.')


def selection_score(values: Mapping[str, float]) -> float:
    """Return the current model-selection score from complete evaluation values."""
    return values[SELECTION_METRIC.key]
