from dataclasses import dataclass
from typing import Self

from src.inference.generation import Generation
from src.logs.declare import ConformanceChecker
from src.metrics import Direction, ScalarRecord, Unit, metric


@dataclass(frozen=True, slots=True)
class ConformanceScores(ScalarRecord):
    """Score for conformance of generated suffixes to process constraints."""

    conformance_sample_mean: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)

    @classmethod
    def of(cls, generation: Generation, *, checker: ConformanceChecker) -> Self:
        """Score generated suffixes against constraints.

        Args:
            generation: Decoded model output for one prefix.
            checker: Process-constraint checker.

        Returns:
            Conformance scores for the prefix.
        """
        # Constraints apply to the complete trace.
        prefix = generation.prefix_activities
        samples = generation.samples

        # Check each distinct suffix once, then weight by draw count.
        checks = [checker.check(prefix + suffix) for suffix in samples.suffixes]
        draws = len(samples)

        def over_draws(values: list[float]) -> float:
            """The mean over the draws of a value read once per distinct suffix."""
            return float(samples.counts @ values) / draws if values and draws else 0.0

        return cls(
            conformance_sample_mean=over_draws([check.share for check in checks]),
        )


@dataclass(frozen=True, slots=True)
class ConformanceDiagnostics(ScalarRecord):
    """Conformance values retained for run diagnostics."""

    conformance_point: float
    conformance_observed: float
    full_conformance_sample_rate: float
    full_conformance_observed: float

    @classmethod
    def of(cls, generation: Generation, *, checker: ConformanceChecker) -> Self:
        """Score point and observed suffixes and the full-constraint sample rate."""
        prefix = generation.prefix_activities
        samples = generation.samples
        checks = [checker.check(prefix + suffix) for suffix in samples.suffixes]
        draws = len(samples)
        full_sample_rate = (
            float(samples.counts @ [check.full for check in checks]) / draws
            if checks and draws
            else 0.0
        )
        return cls(
            conformance_point=checker.check(prefix + generation.point.activities).share,
            conformance_observed=checker.check(prefix + generation.truth.activities).share,
            full_conformance_sample_rate=full_sample_rate,
            full_conformance_observed=checker.check(prefix + generation.truth.activities).full,
        )
