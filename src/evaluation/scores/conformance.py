from dataclasses import dataclass
from typing import Self

from src.inference.generation import Generation
from src.logs.conformance import ConformanceChecker
from src.scalar_metrics import ScalarMetrics, Unit, mean, metric


@dataclass(frozen=True, slots=True)
class ConformanceScores(ScalarMetrics):
    """Whether a prefix's generated suffixes are traces the process allows at all.

    The scores of one prefix's generated samples, or their mean over a set of prefixes.
    """

    # The mean over a prefix's samples
    conformance_mean: float = metric(unit=Unit.SHARE, higher_is_better=True)
    # The suffix written from the mean of `p(z | prefix)`
    conformance_point: float = metric(unit=Unit.SHARE, higher_is_better=True)

    @classmethod
    def of(cls, generation: Generation, *, checker: ConformanceChecker) -> Self:
        """Check one prefix's generated suffixes and its point prediction.

        Args:
            generation: The model's answer for one prefix, decoded into the log's own units.
            checker: The declarative model to check against, held by the process doing the scoring.
        Returns:
            The prefix's conformance. A prefix with no samples scores 0.0 on `conformance_mean`,
            the worst it can be, rather than looking perfectly conformant.
        """
        # A constraint is about the whole trace, so a suffix is checked as the case it completes.
        prefix = tuple(generation.prefix_activities)

        return cls(
            conformance_mean=mean(
                [checker.rate(prefix + tuple(sample.activities)) for sample in generation.samples]
            ),
            conformance_point=checker.rate(prefix + tuple(generation.point.activities)),
        )
