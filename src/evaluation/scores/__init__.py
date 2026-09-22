from src.evaluation.scores import activity, conformance, suffix_length, time
from src.evaluation.scores.context import ScoringContext
from src.evaluation.scores.registry import METRICS, MetricRegistry
from src.metrics import SELECTION_METRIC, Direction

if METRICS[SELECTION_METRIC].direction is not Direction.LOWER:
    raise ValueError(f'{SELECTION_METRIC} must be a minimizing selection metric.')

__all__ = [
    'METRICS',
    'MetricRegistry',
    'ScoringContext',
    'activity',
    'conformance',
    'suffix_length',
    'time',
]
