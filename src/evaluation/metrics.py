from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Self

from Declare4Py.ProcessModels.DeclareModel import DeclareModel

from src.evaluation.accuracy import AccuracyScores, score_generation
from src.evaluation.conformance import ConformanceScores, score_conformance
from src.inference import Generation


@dataclass(frozen=True)
class PairScores:
    """One prefix's scores, or their mean over a set of prefixes.

    Accuracy asks how close a generated suffix is to the one that actually happened;
    conformance asks whether it is a trace the process allows at all.
    """

    accuracy: AccuracyScores
    conformance: ConformanceScores

    @classmethod
    def mean(cls, values: Sequence[Self]) -> Self:
        """Average a set of prefix scores, each family averaged by its own rules.

        Args:
            values: The scores to average, one per prefix.
        Returns:
            The mean accuracy beside the mean conformance.
        """
        return cls(
            accuracy=AccuracyScores.mean([value.accuracy for value in values]),
            conformance=ConformanceScores.mean([value.conformance for value in values]),
        )


@dataclass(frozen=True)
class ScoredPrefix:
    """One prefix's identity beside its scores, the unit a worker hands back."""

    case_id: str
    prefix_len: int
    suffix_len: int
    samples: int
    scores: PairScores


@dataclass(frozen=True)
class ByLengthMetrics:
    """The scores of the prefixes sharing one length, and how many pairs that length had."""

    length: int
    pairs_count: int
    scores: PairScores


def _by_length(buckets: dict[int, list[PairScores]]) -> list[ByLengthMetrics]:
    """Summarize buckets of scores keyed by length, in increasing order of length.

    Args:
        buckets: Scores grouped by some length, e.g. prefix length or suffix length.
    Returns:
        One entry per length present in `buckets`, sorted by length.
    """
    return [
        ByLengthMetrics(
            length=length,
            pairs_count=len(buckets[length]),
            scores=PairScores.mean(buckets[length]),
        )
        for length in sorted(buckets)
    ]


@dataclass(frozen=True)
class EvaluationMetrics:
    """The scores of all prefixes, their breakdown by length, and the population
    they were taken over."""

    pairs: int
    cases: int
    samples_per_prefix: int

    scores: PairScores
    # In increasing order of prefix length
    by_prefix_length: list[ByLengthMetrics]
    # In increasing order of ground-truth suffix length
    by_suffix_length: list[ByLengthMetrics]

    @classmethod
    def aggregate(cls, scored: Iterable[ScoredPrefix]) -> Self:
        """Fold every prefix's scores into the averages one evaluation reports.

        Args:
            scored: The scores of each prefix, in any order and read in a single pass, so they can
                be streamed in as they are computed rather than held all at once.
        Returns:
            The averages over every prefix, the same averages broken down by cut point and by
            ground-truth suffix length, and the population they were taken over. Every prefix
            weighs the same however many samples were drawn for it, so a prefix is the unit this
            describes and a sample is not.
        """
        prefix_buckets: dict[int, list[PairScores]] = {}
        suffix_buckets: dict[int, list[PairScores]] = {}
        cases: set[str] = set()
        samples_per_prefix = 0

        for prefix in scored:
            cases.add(prefix.case_id)
            # Every prefix is drawn for the same number of times, so the first answers for all.
            samples_per_prefix = samples_per_prefix or prefix.samples
            # Bucket the scores by prefix length and by ground-truth suffix length
            prefix_buckets.setdefault(prefix.prefix_len, []).append(prefix.scores)
            suffix_buckets.setdefault(prefix.suffix_len, []).append(prefix.scores)

        # Flatten the buckets into a single list of scores to compute the overall mean
        every_prefix = [scores for bucket in prefix_buckets.values() for scores in bucket]
        return cls(
            pairs=len(every_prefix),
            cases=len(cases),
            samples_per_prefix=samples_per_prefix,
            scores=PairScores.mean(every_prefix),
            by_prefix_length=_by_length(prefix_buckets),
            by_suffix_length=_by_length(suffix_buckets),
        )


def score_prefixes(
    generations: Iterable[Generation],
    *,
    declare_model: DeclareModel,
    consider_vacuity: bool,
) -> list[ScoredPrefix]:
    """Score generated suffixes against the ground truth they were generated for, and against the
    declarative model the dataset was mined for.

    Args:
        generations: The model's answers, one per prefix, decoded into the log's own units.
        declare_model: The declarative model to check conformance against, from
            `load_declare_model`.
        consider_vacuity: Whether a constraint a trace never activates counts as satisfied.
    Returns:
        One entry per prefix, in the order they were read. Only what `EvaluationMetrics.aggregate`
        needs travels back, the generations themselves being dropped here.
    """
    return [
        ScoredPrefix(
            case_id=generation.case_id,
            prefix_len=generation.prefix_len,
            suffix_len=len(generation.truth),
            samples=len(generation.samples),
            scores=PairScores(
                accuracy=score_generation(generation),
                conformance=score_conformance(
                    generation,
                    model=declare_model,
                    consider_vacuity=consider_vacuity,
                ),
            ),
        )
        for generation in generations
    ]
