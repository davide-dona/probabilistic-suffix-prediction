from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from src.evaluation.scores import METRICS, ScoringContext
from src.inference.generation import Generation
from src.logs.declare import ConformanceChecker
from src.metrics import MetricGroup

GROUPS = tuple(MetricGroup)


@dataclass(frozen=True)
class ScoreGroups:
    """Every metric value for a prefix or an aggregate, grouped by evaluation question."""

    activity: dict[str, float]
    suffix_length: dict[str, float]
    time: dict[str, float]
    conformance: dict[str, float]

    @classmethod
    def of(cls, values: dict[str, float]) -> 'ScoreGroups':
        """Group a complete mapping of registered metric values."""
        expected = set(METRICS.entries)
        missing = expected - set(values)
        extra = set(values) - expected
        if missing or extra:
            raise ValueError(
                'metric values differ from the registry: '
                f'missing {sorted(missing)}, extra {sorted(extra)}.'
            )
        grouped = {
            group: {
                key: values[key] for key, metric in METRICS.entries.items() if metric.group is group
            }
            for group in GROUPS
        }
        return cls(**grouped)

    @classmethod
    def mean(cls, values: Sequence['ScoreGroups']) -> 'ScoreGroups':
        """Average complete score mappings, field by field."""
        return cls.of(
            {
                key: sum(value.flatten()[key] for value in values) / len(values) if values else 0.0
                for key in METRICS.entries
            }
        )

    def flatten(self) -> dict[str, float]:
        """Return all values in registry declaration order."""
        groups = {
            MetricGroup.ACTIVITY: self.activity,
            MetricGroup.SUFFIX_LENGTH: self.suffix_length,
            MetricGroup.TIME: self.time,
            MetricGroup.CONFORMANCE: self.conformance,
        }
        return {key: groups[metric.group][key] for key, metric in METRICS.entries.items()}


@dataclass(frozen=True)
class PrefixSummary:
    """Scores for one generated prefix, returned by a worker."""

    prefix_len: int
    suffix_len: int
    scores: ScoreGroups

    @classmethod
    def of(cls, generation: Generation, *, checker: ConformanceChecker) -> 'PrefixSummary':
        """Score one generated suffix against truth and constraints."""
        context = ScoringContext.of(generation, checker=checker)
        return cls(
            prefix_len=generation.prefix_len,
            suffix_len=len(generation.truth),
            scores=ScoreGroups.of(
                {key: metric.compute(context) for key, metric in METRICS.entries.items()}
            ),
        )


@dataclass(frozen=True)
class LengthSummary:
    """Mean scores for prefixes with one shared length."""

    length: int
    prefixes: int
    activity: dict[str, float]
    suffix_length: dict[str, float]
    time: dict[str, float]
    conformance: dict[str, float]

    @classmethod
    def of(cls, prefixes: Sequence[PrefixSummary], *, length: int) -> 'LengthSummary':
        """Aggregate scores for prefixes with a common length."""
        scores = ScoreGroups.mean([prefix.scores for prefix in prefixes])
        return cls(
            length=length,
            prefixes=len(prefixes),
            activity=scores.activity,
            suffix_length=scores.suffix_length,
            time=scores.time,
            conformance=scores.conformance,
        )


def _by_length(buckets: dict[int, list[PrefixSummary]]) -> list[LengthSummary]:
    """Summarize length buckets in ascending order."""
    return [LengthSummary.of(buckets[length], length=length) for length in sorted(buckets)]


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate evaluation scores for one run."""

    prefixes: int
    activity: dict[str, float]
    suffix_length: dict[str, float]
    time: dict[str, float]
    conformance: dict[str, float]
    by_prefix_length: list[LengthSummary]
    by_suffix_length: list[LengthSummary]

    @classmethod
    def of(cls, prefixes: Iterable[PrefixSummary]) -> 'EvaluationSummary':
        """Aggregate prefix scores overall and by prefix and suffix length."""
        prefix_buckets: dict[int, list[PrefixSummary]] = {}
        suffix_buckets: dict[int, list[PrefixSummary]] = {}
        for prefix in prefixes:
            prefix_buckets.setdefault(prefix.prefix_len, []).append(prefix)
            suffix_buckets.setdefault(prefix.suffix_len, []).append(prefix)
        every_prefix = [prefix for bucket in prefix_buckets.values() for prefix in bucket]
        scores = ScoreGroups.mean([prefix.scores for prefix in every_prefix])
        return cls(
            prefixes=len(every_prefix),
            activity=scores.activity,
            suffix_length=scores.suffix_length,
            time=scores.time,
            conformance=scores.conformance,
            by_prefix_length=_by_length(prefix_buckets),
            by_suffix_length=_by_length(suffix_buckets),
        )


type Summarized = PrefixSummary | LengthSummary | EvaluationSummary


def flatten_scores(summary: Summarized) -> dict[str, float]:
    """Flatten a summary's grouped scores into a registry-ordered mapping."""
    if isinstance(summary, PrefixSummary):
        return summary.scores.flatten()
    return ScoreGroups(
        activity=summary.activity,
        suffix_length=summary.suffix_length,
        time=summary.time,
        conformance=summary.conformance,
    ).flatten()
