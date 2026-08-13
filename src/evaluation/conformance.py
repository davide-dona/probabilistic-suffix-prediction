from dataclasses import dataclass

from src.inference.generation import Generation
from src.logs.declare import ConformanceChecker
from src.metrics import ScalarMetrics, mean


@dataclass(frozen=True, slots=True)
class ConformanceScores(ScalarMetrics):
    """The scores of one prefix's generated samples."""

    conformance_mean: float  # the mean over a prefix's samples
    conformance_point: float  # the suffix written from the mean of `p(z | prefix)`


def score_conformance(
    generation: Generation,
    *,
    checker: ConformanceChecker,
) -> ConformanceScores:
    """
    Check one prefix's generated suffixes and its point prediction.

    Args:
        generation: The model's answer for one prefix, decoded into the log's own units.
        checker: The declarative model to check against, held by the process doing the scoring.
    Returns:
        The prefix's conformance. A prefix with no samples scores 0.0 on `conformance_mean`,
        the worst it can be, rather than looking perfectly conformant.
    """
    # A constraint is about the whole trace, so a suffix is checked as the case it completes.
    prefix = tuple(generation.prefix_activities)

    return ConformanceScores(
        conformance_mean=mean(
            [checker.rate(prefix + tuple(sample.activities)) for sample in generation.samples]
        ),
        conformance_point=checker.rate(prefix + tuple(generation.point.activities)),
    )
