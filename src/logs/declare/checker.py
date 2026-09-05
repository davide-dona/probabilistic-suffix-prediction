from dataclasses import replace
from functools import lru_cache

from src import paths
from src.logs.declare.constraints import read_constraints
from src.logs.declare.templates import Positions
from src.suffixes import ActivityCodes

# Stands in for an activity the dataset's codebook does not know, so its constraint can never be
# activated by a trace the codebook spelled.
_UNMATCHABLE = '\x00'


class ConformanceChecker:
    """Scores traces against the declarative model a dataset was mined for."""

    def __init__(self, dataset: str, codes: ActivityCodes) -> None:
        """
        Args:
            dataset: The dataset whose model to check against, read from where preprocessing
                wrote it.
            codes: The dataset's codebook, which the constraints are translated onto so a trace is
                checked as the string the generations already hold it as, with nothing decoded per
                check. An activity the codebook does not know is given a character no trace can
                contain, leaving its constraint unactivated rather than growing the codebook.
        """
        self._constraints = tuple(
            replace(
                constraint,
                first=codes.codes.get(constraint.first, _UNMATCHABLE),
                second=(
                    None
                    if constraint.second is None
                    else codes.codes.get(constraint.second, _UNMATCHABLE)
                ),
            )
            for constraint in read_constraints(paths.DECLARE_MODEL.require(dataset))
        )

    @lru_cache(maxsize=100_000)  # noqa: B019 -- one checker per scoring process
    def rate(self, trace: str) -> float:
        """
        The fraction of the model's constraints one trace satisfies.

        Args:
            trace: The trace's activities, one character each, in order, on the dataset's own
                scale. A whole case, prefix included: a constraint like `Init` or `Precedence` is
                about the trace, not about a run of events inside it.
        Returns:
            The satisfied share, in `[0, 1]`, or 0.0 for a model that checks nothing.
        """
        positions: Positions = {}
        for index, activity in enumerate(trace):
            positions.setdefault(activity, []).append(index)

        satisfied = sum(constraint.holds(trace, positions) for constraint in self._constraints)
        return satisfied / len(self._constraints) if self._constraints else 0.0
