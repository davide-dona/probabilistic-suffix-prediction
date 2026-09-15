from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields
from typing import Self

from src.evaluation.scores import (
    FAMILIES,
    CalibrationScores,
    ConformanceScores,
    PointPredictionScores,
    SamplePredictionScores,
    ScoringContext,
)
from src.inference.generation import Generation
from src.logs.declare import ConformanceChecker


@dataclass(frozen=True, slots=True)
class PrefixSummary:
    """Scores for one prefix, returned by a worker."""

    prefix_len: int
    suffix_len: int
    point: PointPredictionScores
    sample: SamplePredictionScores
    calibration: CalibrationScores
    conformance: ConformanceScores

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
            point=PointPredictionScores.of(context),
            sample=SamplePredictionScores.of(context),
            calibration=CalibrationScores.of(context),
            conformance=ConformanceScores.of(generation, checker=checker),
        )


@dataclass(frozen=True)
class LengthSummary:
    """Mean scores for prefixes with one shared length."""

    length: int
    prefixes: int
    point: PointPredictionScores
    sample: SamplePredictionScores
    calibration: CalibrationScores
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
            point=PointPredictionScores.mean([prefix.point for prefix in prefixes]),
            sample=SamplePredictionScores.mean([prefix.sample for prefix in prefixes]),
            calibration=CalibrationScores.mean([prefix.calibration for prefix in prefixes]),
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
    point: PointPredictionScores
    sample: SamplePredictionScores
    calibration: CalibrationScores
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
            point=PointPredictionScores.mean([prefix.point for prefix in every_prefix]),
            sample=SamplePredictionScores.mean([prefix.sample for prefix in every_prefix]),
            calibration=CalibrationScores.mean(
                [prefix.calibration for prefix in every_prefix]
            ),
            conformance=ConformanceScores.mean([prefix.conformance for prefix in every_prefix]),
            by_prefix_length=_by_length(prefix_buckets),
            by_suffix_length=_by_length(suffix_buckets),
        )


# Types that expose all score families.
type Summarized = PrefixSummary | LengthSummary | EvaluationSummary

# Cache each family's metric fields.
_FIELD_NAMES = {family: tuple(entry.name for entry in fields(family)) for family in FAMILIES}


def flatten_scores(summary: Summarized) -> dict[str, float]:
    """Flatten a summary's score families into a metric mapping.

    Args:
        summary: Prefix, length, or evaluation aggregate.

    Returns:
        Metric values keyed by metric name.
    """
    return {
        name: getattr(family, name)
        for family in (
            summary.point,
            summary.sample,
            summary.calibration,
            summary.conformance,
        )
        for name in _FIELD_NAMES[type(family)]
    }
