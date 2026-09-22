from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields
from typing import Self

from src.evaluation.scores import (
    FAMILIES,
    ActivityDiagnostics,
    ActivityScores,
    ConformanceDiagnostics,
    ConformanceScores,
    ScoringContext,
    SuffixLengthDiagnostics,
    SuffixLengthScores,
    TimeDiagnostics,
)
from src.inference.generation import Generation
from src.logs.declare import ConformanceChecker


@dataclass(frozen=True, slots=True)
class PrefixSummary:
    """Scores for one prefix, returned by a worker."""

    prefix_len: int
    suffix_len: int
    activity: ActivityScores
    suffix_length: SuffixLengthScores
    conformance: ConformanceScores
    activity_diagnostics: ActivityDiagnostics
    suffix_length_diagnostics: SuffixLengthDiagnostics
    time_diagnostics: TimeDiagnostics
    conformance_diagnostics: ConformanceDiagnostics

    @classmethod
    def of(
        cls,
        generation: Generation,
        *,
        checker: ConformanceChecker,
    ) -> Self:
        """Score one generated suffix against truth and constraints.

        Args:
            generation: Decoded model output for one prefix.
            checker: Process-constraint checker.

        Returns:
            Scores for the prefix.
        """
        context = ScoringContext.of(generation)
        return cls(
            prefix_len=generation.prefix_len,
            suffix_len=len(generation.truth),
            activity=ActivityScores.of(context),
            suffix_length=SuffixLengthScores.of(context),
            conformance=ConformanceScores.of(generation, checker=checker),
            activity_diagnostics=ActivityDiagnostics.of(context),
            suffix_length_diagnostics=SuffixLengthDiagnostics.of(context),
            time_diagnostics=TimeDiagnostics.of(context),
            conformance_diagnostics=ConformanceDiagnostics.of(generation, checker=checker),
        )


@dataclass(frozen=True)
class LengthSummary:
    """Mean scores for prefixes with one shared length."""

    length: int
    prefixes: int
    activity: ActivityScores
    suffix_length: SuffixLengthScores
    conformance: ConformanceScores

    @classmethod
    def of(cls, prefixes: Sequence[PrefixSummary], *, length: int) -> Self:
        """Aggregate scores for prefixes with a common length.

        Args:
            prefixes: Prefix summaries sharing the length.
            length: Shared prefix or suffix length.

        Returns:
            Aggregate scores for the length.
        """
        return cls(
            length=length,
            prefixes=len(prefixes),
            activity=ActivityScores.mean([prefix.activity for prefix in prefixes]),
            suffix_length=SuffixLengthScores.mean([prefix.suffix_length for prefix in prefixes]),
            conformance=ConformanceScores.mean([prefix.conformance for prefix in prefixes]),
        )


def _by_length(buckets: dict[int, list[PrefixSummary]]) -> list[LengthSummary]:
    """Summarize length buckets in ascending order.

    Args:
        buckets: Prefix summaries grouped by length.

    Returns:
        One aggregate per length.
    """
    return [LengthSummary.of(buckets[length], length=length) for length in sorted(buckets)]


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate evaluation scores for one run."""

    prefixes: int
    activity: ActivityScores
    suffix_length: SuffixLengthScores
    conformance: ConformanceScores
    # Sorted by prefix length.
    by_prefix_length: list[LengthSummary]
    # Sorted by ground-truth suffix length.
    by_suffix_length: list[LengthSummary]

    @classmethod
    def of(cls, prefixes: Iterable[PrefixSummary]) -> Self:
        """Aggregate prefix scores overall and by prefix and suffix length.

        Args:
            prefixes: Prefix summaries to aggregate.

        Returns:
            Overall and length-bucketed evaluation scores.
        """
        prefix_buckets: dict[int, list[PrefixSummary]] = {}
        suffix_buckets: dict[int, list[PrefixSummary]] = {}

        for prefix in prefixes:
            # Group by both reported length axes.
            prefix_buckets.setdefault(prefix.prefix_len, []).append(prefix)
            suffix_buckets.setdefault(prefix.suffix_len, []).append(prefix)

        # Recover all prefixes for the overall aggregate.
        every_prefix = [prefix for bucket in prefix_buckets.values() for prefix in bucket]
        return cls(
            prefixes=len(every_prefix),
            activity=ActivityScores.mean([prefix.activity for prefix in every_prefix]),
            suffix_length=SuffixLengthScores.mean(
                [prefix.suffix_length for prefix in every_prefix]
            ),
            conformance=ConformanceScores.mean([prefix.conformance for prefix in every_prefix]),
            by_prefix_length=_by_length(prefix_buckets),
            by_suffix_length=_by_length(suffix_buckets),
        )


type Summarized = PrefixSummary | LengthSummary | EvaluationSummary

_FIELD_NAMES = {family: tuple(entry.name for entry in fields(family)) for family in FAMILIES}


def flatten_scores(summary: Summarized) -> dict[str, float]:
    """Flatten a summary's reported score families into one metric mapping.

    Args:
        summary: Prefix, length, or evaluation aggregate.
    Returns:
        Metric values keyed by their declared field names.
    """
    return {
        name: getattr(family, name)
        for family in (summary.activity, summary.suffix_length, summary.conformance)
        for name in _FIELD_NAMES[type(family)]
    }
