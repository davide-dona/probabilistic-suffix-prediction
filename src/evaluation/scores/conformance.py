from dataclasses import dataclass
from typing import Self

from src.inference.generation import Generation
from src.logs.conformance import ConformanceChecker
from src.scalar_metrics import Direction, ScalarMetrics, Unit, mean, metric


@dataclass(frozen=True, slots=True)
class ConformanceScores(ScalarMetrics):
    """Whether a prefix's generated suffixes are traces the process allows at all.

    The scores of one prefix's generated samples and of the suffix it actually took, or their mean
    over a set of prefixes.
    """

    # The mean over a prefix's samples
    conformance_mean: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)
    # The suffix written from the mean of `p(z | prefix)`
    conformance_point: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)
    # The suffix the log actually took, through the same checker. A property of the log rather
    # than of the model, so it is the same for every model of a log and is never compared between
    # them: it is the target the two levels above are read against.
    conformance_truth: float = metric(unit=Unit.SHARE)

    # Each of the two read against the suffix the log actually took. A level alone says little:
    # a model that always emitted the log's most common variant would score near 1.0, since real
    # traces are what carry the rare deviations. Positive means the generated suffixes are more
    # conformant than the ones the process produced, which is as wrong as being less conformant,
    # so the target is 0 and the best of a row is the one nearest it.
    conformance_gap_mean: float = metric(unit=Unit.SCORE, direction=Direction.ZERO)
    conformance_gap_point: float = metric(unit=Unit.SCORE, direction=Direction.ZERO)

    @classmethod
    def of(cls, generation: Generation, *, checker: ConformanceChecker) -> Self:
        """Check one prefix's generated suffixes, its point prediction and its ground truth.

        Args:
            generation: The model's answer for one prefix, decoded into the log's own units.
            checker: The declarative model to check against, held by the process doing the scoring.
        Returns:
            The prefix's conformance, that of the suffix the log took, and each of the first two
            read against it. A prefix with no samples scores 0.0 on `conformance_mean`, the worst
            it can be, rather than looking perfectly conformant.
        """
        # A constraint is about the whole trace, so a suffix is checked as the case it completes.
        prefix = tuple(generation.prefix_activities)

        sampled = mean(
            [checker.rate(prefix + tuple(sample.activities)) for sample in generation.samples]
        )
        point = checker.rate(prefix + tuple(generation.point.activities))
        truth = checker.rate(prefix + tuple(generation.truth.activities))

        return cls(
            conformance_mean=sampled,
            conformance_point=point,
            conformance_truth=truth,
            # A mean of differences is the difference of the means, so a gap column and the two
            # level columns it is read from stay consistent at every granularity.
            conformance_gap_mean=sampled - truth,
            conformance_gap_point=point - truth,
        )
