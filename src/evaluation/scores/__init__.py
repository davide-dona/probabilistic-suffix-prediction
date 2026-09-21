from src.evaluation.scores.activity import ActivityDiagnostics, ActivityScores
from src.evaluation.scores.conformance import ConformanceDiagnostics, ConformanceScores
from src.evaluation.scores.context import ScoringContext
from src.evaluation.scores.suffix_length import SuffixLengthDiagnostics, SuffixLengthScores
from src.evaluation.scores.time import TimeDiagnostics
from src.registry import Registry
from src.scalar_metrics import Metric

# Score families in report order.
FAMILIES = (
    ActivityScores,
    SuffixLengthScores,
    ConformanceScores,
)


def _declared() -> dict[str, Metric]:
    """Collect score declarations, rejecting duplicate names.

    Returns:
        Metric declarations keyed by field name.

    Raises:
        ValueError: If score families reuse a field name.
    """
    entries: dict[str, Metric] = {}
    for family in FAMILIES:
        for declaration in family.metrics():
            if declaration.key in entries:
                raise ValueError(
                    f'{declaration.key} is declared twice. A report holds every score in one '
                    f'namespace, so a name belongs to a single field.'
                )
            entries[declaration.key] = declaration
    return entries


# Score metadata assembled from the declared family fields.
METRICS = Registry[Metric](
    kind='metric',
    where='the score fields of the families in src/evaluation/scores/',
    entries=_declared(),
)

__all__ = [
    'FAMILIES',
    'METRICS',
    'ActivityDiagnostics',
    'ActivityScores',
    'ConformanceDiagnostics',
    'ConformanceScores',
    'ScoringContext',
    'SuffixLengthDiagnostics',
    'SuffixLengthScores',
    'TimeDiagnostics',
]
