from src.evaluation.metrics import activity, conformance, suffix_length, time
from src.evaluation.metrics.prepared import PreparedPrefix
from src.evaluation.metrics.registry import METRICS, MetricRegistry
from src.metrics import SELECTION_METRIC, Direction

if METRICS[SELECTION_METRIC].direction is not Direction.LOWER:
    raise ValueError(f'{SELECTION_METRIC} must be a minimizing selection metric.')

__all__ = [
    'METRICS',
    'MetricRegistry',
    'PreparedPrefix',
    'activity',
    'conformance',
    'suffix_length',
    'time',
]
