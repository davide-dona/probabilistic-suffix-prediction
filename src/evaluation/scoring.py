from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from src.evaluation.metrics import METRICS
from src.evaluation.metrics.metadata import MetricGroup
from src.evaluation.prepared import PreparedPrefix
from src.inference.generation import Generation
from src.logs.declare import ConformanceChecker


@dataclass(frozen=True)
class ScoreGroups:
    """Metric values grouped by evaluation question."""

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
        grouped: dict[str, dict[str, float]] = {group: {} for group in MetricGroup}
        for key, metric in METRICS.entries.items():
            grouped[metric.group][key] = values[key]
        return cls(**grouped)

    @classmethod
    def mean(cls, values: Sequence['ScoreGroups']) -> 'ScoreGroups':
        """Average scores, returning zero for each metric when empty."""
        accumulator = _ScoreAccumulator()
        for scores in values:
            accumulator.add(scores.flatten())
        return accumulator.mean()

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
        context = PreparedPrefix.of(generation, checker=checker)
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
    scores: ScoreGroups


@dataclass
class _ScoreAccumulator:
    totals: dict[str, float] = field(default_factory=lambda: dict.fromkeys(METRICS.entries, 0.0))
    count: int = 0

    def add(self, values: Mapping[str, float]) -> None:
        for key in self.totals:
            self.totals[key] += values[key]
        self.count += 1

    def mean(self) -> ScoreGroups:
        if self.count == 0:
            return ScoreGroups.of(self.totals.copy())
        return ScoreGroups.of({key: total / self.count for key, total in self.totals.items()})

    def length_summary(self, length: int) -> LengthSummary:
        return LengthSummary(length=length, prefixes=self.count, scores=self.mean())


def _add_to_bucket(
    buckets: dict[int, _ScoreAccumulator], length: int, values: Mapping[str, float]
) -> None:
    if length not in buckets:
        buckets[length] = _ScoreAccumulator()
    buckets[length].add(values)


def _by_length(buckets: dict[int, _ScoreAccumulator]) -> list[LengthSummary]:
    return [buckets[length].length_summary(length) for length in sorted(buckets)]


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate evaluation scores for one run."""

    prefixes: int
    scores: ScoreGroups
    by_prefix_length: list[LengthSummary]
    by_suffix_length: list[LengthSummary]

    @classmethod
    def of(cls, prefixes: Iterable[PrefixSummary]) -> 'EvaluationSummary':
        """Aggregate a single pass over prefix scores with equal prefix weights."""
        overall = _ScoreAccumulator()
        prefix_buckets: dict[int, _ScoreAccumulator] = {}
        suffix_buckets: dict[int, _ScoreAccumulator] = {}
        for prefix in prefixes:
            values = prefix.scores.flatten()
            overall.add(values)
            _add_to_bucket(prefix_buckets, prefix.prefix_len, values)
            _add_to_bucket(suffix_buckets, prefix.suffix_len, values)
        return cls(
            prefixes=overall.count,
            scores=overall.mean(),
            by_prefix_length=_by_length(prefix_buckets),
            by_suffix_length=_by_length(suffix_buckets),
        )
