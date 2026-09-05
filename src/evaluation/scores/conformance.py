from dataclasses import dataclass
from typing import Self

from src.inference.generation import Generation
from src.logs.declare import ConformanceChecker
from src.scalar_metrics import Direction, Owner, ScalarMetrics, Unit, metric


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
    conformance_truth: float = metric(unit=Unit.SHARE, owner=Owner.LOG)

    @classmethod
    def of(cls, generation: Generation, *, checker: ConformanceChecker) -> Self:
        """Check one prefix's generated suffixes, its point prediction and its ground truth.

        Args:
            generation: The model's answer for one prefix, decoded into the log's own units.
            checker: The declarative model to check against, held by the process doing the scoring.
        Returns:
            The prefix's conformance and that of the suffix the log took, which is the target the
            two levels are read against. A prefix with no samples scores 0.0 on
            `conformance_mean`, the worst it can be, rather than looking perfectly conformant.
        """
        # A constraint is about the whole trace, so a suffix is checked as the case it completes.
        prefix = generation.prefix_activities
        samples = generation.samples

        # One check per distinct suffix, weighed by how many draws took it. A repeated draw is the
        # same trace and rates the same, so this is the mean over the draws with the constraints
        # walked once per distinct suffix rather than once per draw.
        rates = [checker.rate(prefix + suffix) for suffix in samples.suffixes]
        draws = len(samples)

        return cls(
            conformance_mean=float(samples.counts @ rates) / draws if rates and draws else 0.0,
            conformance_point=checker.rate(prefix + generation.point.activities),
            conformance_truth=checker.rate(prefix + generation.truth.activities),
        )
