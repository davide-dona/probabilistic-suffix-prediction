from dataclasses import fields

from src.evaluation.scores.activity import ActivityDiagnostics, ActivityScores
from src.evaluation.scores.conformance import ConformanceDiagnostics, ConformanceScores
from src.evaluation.scores.context import ScoringContext
from src.evaluation.scores.suffix_length import SuffixLengthDiagnostics, SuffixLengthScores
from src.evaluation.scores.time import TimeDiagnostics
from src.metrics import Metric, metrics_of
from src.registry import Registry

FAMILIES = (ActivityScores, SuffixLengthScores, ConformanceScores)


def _declared() -> dict[str, Metric]:
    """Collect each reported score once, rejecting missing and repeated declarations."""
    entries: dict[str, Metric] = {}
    for family in FAMILIES:
        declared = metrics_of(family)
        keys = {metric.key for metric in declared}
        missing = [entry.name for entry in fields(family) if entry.name not in keys]
        if missing:
            raise ValueError(f'{family.__name__} has undeclared scores: {", ".join(missing)}.')
        for metric in declared:
            if metric.key in entries:
                raise ValueError(f'{metric.key} is declared by more than one score family.')
            entries[metric.key] = metric
    return entries


METRICS = Registry[Metric](
    kind='metric',
    where='the score fields in src/evaluation/scores/',
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
