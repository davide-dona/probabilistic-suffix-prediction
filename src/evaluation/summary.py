from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields
from typing import Self

from src.evaluation.scores import FAMILIES, AccuracyScores, ConformanceScores, DistributionScores
from src.inference.generation import Generation
from src.logs.conformance import ConformanceChecker
from src.logs.continuations import ContinuationIndex


@dataclass(frozen=True, slots=True)
class PrefixSummary:
    """One prefix's samples reduced to three families of scores, the unit a worker hands back."""

    prefix_len: int
    suffix_len: int
    accuracy: AccuracyScores
    conformance: ConformanceScores
    distribution: DistributionScores

    @classmethod
    def of(
        cls,
        generation: Generation,
        *,
        checker: ConformanceChecker,
        index: ContinuationIndex,
    ) -> Self:
        """Score one prefix's generated suffixes against the ground truth they were generated for,
        against the declarative model the dataset was mined for, and against every continuation
        the log took after that prefix.

        Args:
            generation: The model's answer for one prefix, decoded into the log's own units.
            checker: The declarative model to check conformance against, held by the process doing
                the scoring.
            index: The log's observed continuations, held by the same process.
        Returns:
            Only what `EvaluationSummary.of` needs, the generation itself being dropped here so
            that a split of hundreds of thousands of prefixes never brings more than a block's
            objects back from a worker.
        """
        return cls(
            prefix_len=generation.prefix_len,
            suffix_len=len(generation.truth),
            accuracy=AccuracyScores.of(generation),
            conformance=ConformanceScores.of(generation, checker=checker),
            distribution=DistributionScores.of(generation, index=index),
        )


@dataclass(frozen=True)
class LengthSummary:
    """The mean scores of the prefixes sharing one length, and how many of them there were."""

    length: int
    prefixes: int
    accuracy: AccuracyScores
    conformance: ConformanceScores
    distribution: DistributionScores

    @classmethod
    def of(cls, prefixes: Sequence[PrefixSummary], *, length: int) -> Self:
        """Average the prefixes of one length, each family by its own rules.

        Args:
            prefixes: The prefixes sharing that length, in any order.
            length: What they share, whether a cut point or a ground-truth suffix length.
        Returns:
            Their means, beside how many of them there were.
        """
        return cls(
            length=length,
            prefixes=len(prefixes),
            accuracy=AccuracyScores.mean([prefix.accuracy for prefix in prefixes]),
            conformance=ConformanceScores.mean([prefix.conformance for prefix in prefixes]),
            distribution=DistributionScores.mean([prefix.distribution for prefix in prefixes]),
        )


def _by_length(buckets: dict[int, list[PrefixSummary]]) -> list[LengthSummary]:
    """Summarize buckets of prefixes keyed by length, in increasing order of length.

    Args:
        buckets: Prefixes grouped by some length, e.g. prefix length or suffix length.
    Returns:
        One entry per length present in `buckets`, sorted by length.
    """
    return [LengthSummary.of(buckets[length], length=length) for length in sorted(buckets)]


@dataclass(frozen=True)
class EvaluationSummary:
    """Everything one evaluation measured over one run's generated suffixes.

    Accuracy asks how close a generated suffix is to the one that actually happened, conformance
    whether it is a trace the process allows at all, distribution how the whole set of suffixes
    generated for a prefix compares against every continuation the log took after it, and all
    three are means over every prefix.

    The two breakdowns cut the per-prefix families either at the prefix or at the ground-truth
    suffix. They are not independent of each other: every cut point of a case is scored, so a long
    prefix is what leaves a short suffix.

    A prefix is the unit throughout. It weighs the same however many samples were drawn for it,
    and a case contributes one prefix per cut point, so a long case weighs more than a short one
    in every mean here; the breakdowns are what read past that.
    """

    prefixes: int
    accuracy: AccuracyScores
    conformance: ConformanceScores
    distribution: DistributionScores
    # In increasing order of prefix length
    by_prefix_length: list[LengthSummary]
    # In increasing order of ground-truth suffix length
    by_suffix_length: list[LengthSummary]

    @classmethod
    def of(cls, prefixes: Iterable[PrefixSummary]) -> Self:
        """Fold every prefix's scores into the averages one evaluation reports.

        Args:
            prefixes: The summary of each prefix, in any order and read in a single pass, so they
                can be streamed in from the pool as they are scored rather than read twice.
        Returns:
            The averages over every prefix, the same averages broken down by cut point and by
            ground-truth suffix length, and how many prefixes they were taken over. Every prefix
            weighs the same however many samples were drawn for it, so a prefix is the unit this
            describes and a sample is not.
        """
        prefix_buckets: dict[int, list[PrefixSummary]] = {}
        suffix_buckets: dict[int, list[PrefixSummary]] = {}

        for prefix in prefixes:
            # Bucket by prefix length and by ground-truth suffix length
            prefix_buckets.setdefault(prefix.prefix_len, []).append(prefix)
            suffix_buckets.setdefault(prefix.suffix_len, []).append(prefix)

        # Flatten one set of buckets back out to take the mean over every prefix at once
        every_prefix = [prefix for bucket in prefix_buckets.values() for prefix in bucket]
        return cls(
            prefixes=len(every_prefix),
            accuracy=AccuracyScores.mean([prefix.accuracy for prefix in every_prefix]),
            conformance=ConformanceScores.mean([prefix.conformance for prefix in every_prefix]),
            distribution=DistributionScores.mean([prefix.distribution for prefix in every_prefix]),
            by_prefix_length=_by_length(prefix_buckets),
            by_suffix_length=_by_length(suffix_buckets),
        )


# The three granularities that hold the three families side by side, and so the three a reader can
# ask for every score of at once.
type Summarized = PrefixSummary | LengthSummary | EvaluationSummary

# The float fields of each family, read once rather than per summary: this is asked a quarter of a
# million times when a run's prefixes are written out one by one.
_FIELD_NAMES = {family: tuple(entry.name for entry in fields(family)) for family in FAMILIES}


def flatten_scores(summary: Summarized) -> dict[str, float]:
    """Every per-prefix metric one summary holds, keyed by its own name.

    Args:
        summary: One prefix's scores, the mean of the prefixes sharing one length, or an
            evaluation's means.
    Returns:
        Every family in one namespace, since a metric's name is unique across them and a figure,
        a table or a column of the per-prefix file asks for a metric rather than for a family.
    """
    return {
        name: getattr(family, name)
        for family in (
            summary.accuracy,
            summary.conformance,
            summary.distribution,
        )
        for name in _FIELD_NAMES[type(family)]
    }
