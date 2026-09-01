import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from src import paths
from src.configs import DeclareConfig

# Parse the declarative model written by `discover_declare_model` back into the constraints it
# holds. Read here rather than through `DeclareModel.parse_from_file`, whose grammar rejects the
# activity names a real log carries: see `read_constraints`.
_CONSTRAINT_LINE = re.compile(r'^(.*)\[(.*)\]\s*(.*)$')
_TEMPLATE_AND_CARDINALITY = re.compile(r'(^.+?)(\d*$)')

# The header `discover_declare_model` opens a model with and `discovery_settings` reads back.
# Comment lines, so nothing that parses the constraints has to know about them.
COMMENT = '#'
SETTINGS_LINE = '# settings: '

# Maximum number of distinct traces cached by the ConformanceChecker. Each prefix generates one
# trace per sample, most of which repeat, and the same trace may be generated for several
# prefixes; caching previously computed scores avoids redundant conformance checks.
_TRACE_CACHE_SIZE = 100_000

# Where each activity of a trace occurs, in order, keyed by its name. Built once per trace, so a
# template answers over the occurrences of the activities it names rather than over the trace.
Positions = dict[str, list[int]]


@dataclass(frozen=True, slots=True)
class Constraint:
    """One constraint of a model: the template it follows and the activities it is about."""

    template: '_Template'
    # The first activity named, which is what activates every template but the precedence family
    first: str
    # The second activity of a binary template, or None for a unary one
    second: str | None
    # How many occurrences of `first` a counting template asks for, and 1 for the rest
    n: int

    def holds(self, trace: tuple[str, ...], positions: Positions) -> bool:
        """Whether one finished trace satisfies this constraint.

        Args:
            trace: The trace's activity names, in order.
            positions: Where each of them occurs, from `ConformanceChecker.rate`.
        Returns:
            True if the trace both activates the constraint and does not violate it.
        """
        return self.template.holds(self, trace, positions)


_Predicate = Callable[[Constraint, tuple[str, ...], Positions], bool]


@dataclass(frozen=True, slots=True)
class _Template:
    """One DECLARE template: what satisfying it means, and how a constraint of it is written."""

    holds: _Predicate
    is_binary: bool
    supports_cardinality: bool


def _existence(constraint: Constraint, trace: tuple[str, ...], positions: Positions) -> bool:
    """`a` occurs at least `n` times."""
    return len(positions.get(constraint.first, ())) >= constraint.n


def _absence(constraint: Constraint, trace: tuple[str, ...], positions: Positions) -> bool:
    """`a` occurs fewer than `n` times, never running it included."""
    return len(positions.get(constraint.first, ())) < constraint.n


def _exactly(constraint: Constraint, trace: tuple[str, ...], positions: Positions) -> bool:
    """`a` occurs exactly `n` times."""
    return len(positions.get(constraint.first, ())) == constraint.n


def _init(constraint: Constraint, trace: tuple[str, ...], positions: Positions) -> bool:
    """`a` is the first event of the trace."""
    return bool(trace) and trace[0] == constraint.first


def _choice(constraint: Constraint, trace: tuple[str, ...], positions: Positions) -> bool:
    """`a` or `b` occurs."""
    return constraint.first in positions or constraint.second in positions


def _responded_existence(
    constraint: Constraint, trace: tuple[str, ...], positions: Positions
) -> bool:
    """`a` occurs, and so does `b`."""
    return constraint.first in positions and constraint.second in positions


def _response(constraint: Constraint, trace: tuple[str, ...], positions: Positions) -> bool:
    """`a` occurs, and every occurrence of it is followed by a `b`."""
    activations = positions.get(constraint.first)
    targets = positions.get(constraint.second)
    return bool(activations) and bool(targets) and activations[-1] < targets[-1]


def _precedence(constraint: Constraint, trace: tuple[str, ...], positions: Positions) -> bool:
    """`b` occurs, and every occurrence of it is preceded by an `a`."""
    activations = positions.get(constraint.second)
    earlier = positions.get(constraint.first)
    return bool(activations) and bool(earlier) and earlier[0] < activations[0]


def _not_response(constraint: Constraint, trace: tuple[str, ...], positions: Positions) -> bool:
    """`a` occurs, and no occurrence of it is followed by a `b`."""
    activations = positions.get(constraint.first)
    targets = positions.get(constraint.second)
    return bool(activations) and (not targets or activations[0] > targets[-1])


def _not_precedence(constraint: Constraint, trace: tuple[str, ...], positions: Positions) -> bool:
    """`b` occurs, and no occurrence of it is preceded by an `a`."""
    activations = positions.get(constraint.second)
    earlier = positions.get(constraint.first)
    return bool(activations) and (not earlier or earlier[0] > activations[-1])


def _chain_response(constraint: Constraint, trace: tuple[str, ...], positions: Positions) -> bool:
    """`a` occurs, and a `b` follows it immediately every time."""
    activations = positions.get(constraint.first)
    last = len(trace) - 1
    return bool(activations) and all(
        index < last and trace[index + 1] == constraint.second for index in activations
    )


def _chain_precedence(constraint: Constraint, trace: tuple[str, ...], positions: Positions) -> bool:
    """`b` occurs, and an `a` precedes it immediately every time."""
    activations = positions.get(constraint.second)
    return bool(activations) and all(
        index > 0 and trace[index - 1] == constraint.first for index in activations
    )


def _not_chain_response(
    constraint: Constraint, trace: tuple[str, ...], positions: Positions
) -> bool:
    """`a` occurs, and a `b` never follows it immediately."""
    activations = positions.get(constraint.first)
    last = len(trace) - 1
    return bool(activations) and not any(
        index < last and trace[index + 1] == constraint.second for index in activations
    )


def _not_chain_precedence(
    constraint: Constraint, trace: tuple[str, ...], positions: Positions
) -> bool:
    """`b` occurs, and an `a` never precedes it immediately."""
    activations = positions.get(constraint.second)
    return bool(activations) and not any(
        index > 0 and trace[index - 1] == constraint.first for index in activations
    )


def _alternate_response(
    constraint: Constraint, trace: tuple[str, ...], positions: Positions
) -> bool:
    """`a` occurs, and a `b` follows each occurrence of it before `a` recurs."""
    activations = fulfillments = 0
    pending = False
    for activity in trace:
        if activity == constraint.first:
            pending = True
            activations += 1
        if pending and activity == constraint.second:
            pending = False
            fulfillments += 1
    return activations > 0 and activations == fulfillments


def _alternate_precedence(
    constraint: Constraint, trace: tuple[str, ...], positions: Positions
) -> bool:
    """`b` occurs, and an `a` precedes each occurrence of it since the previous `b`."""
    activations = fulfillments = 0
    preceding = 0
    for activity in trace:
        if activity == constraint.first:
            preceding += 1
        if activity == constraint.second:
            activations += 1
            if preceding:
                fulfillments += 1
            preceding = 0
    return activations > 0 and activations == fulfillments


# Every template `pipelines.preprocess` can mine, keyed by the name it writes into the model file.
TEMPLATES: dict[str, _Template] = {
    'Existence': _Template(holds=_existence, is_binary=False, supports_cardinality=True),
    'Absence': _Template(holds=_absence, is_binary=False, supports_cardinality=True),
    'Exactly': _Template(holds=_exactly, is_binary=False, supports_cardinality=True),
    'Init': _Template(holds=_init, is_binary=False, supports_cardinality=False),
    'Choice': _Template(holds=_choice, is_binary=True, supports_cardinality=False),
    'Responded Existence': _Template(
        holds=_responded_existence, is_binary=True, supports_cardinality=False
    ),
    'Response': _Template(holds=_response, is_binary=True, supports_cardinality=False),
    'Precedence': _Template(holds=_precedence, is_binary=True, supports_cardinality=False),
    'Not Response': _Template(holds=_not_response, is_binary=True, supports_cardinality=False),
    'Not Precedence': _Template(holds=_not_precedence, is_binary=True, supports_cardinality=False),
    'Chain Response': _Template(holds=_chain_response, is_binary=True, supports_cardinality=False),
    'Chain Precedence': _Template(
        holds=_chain_precedence, is_binary=True, supports_cardinality=False
    ),
    'Not Chain Response': _Template(
        holds=_not_chain_response, is_binary=True, supports_cardinality=False
    ),
    'Not Chain Precedence': _Template(
        holds=_not_chain_precedence, is_binary=True, supports_cardinality=False
    ),
    'Alternate Response': _Template(
        holds=_alternate_response, is_binary=True, supports_cardinality=False
    ),
    'Alternate Precedence': _Template(
        holds=_alternate_precedence, is_binary=True, supports_cardinality=False
    ),
}


def discovery_settings(path: Path) -> DeclareConfig | None:
    """What a declarative model was discovered under, as its own header records it.

    Discovery and conformance checking each decide, separately, whether a constraint a trace never
    activates counts as satisfied: discovery because it decides which constraints hold on enough of
    the log to keep, checking because it decides what a trace is credited for. They are independent
    settings and need not agree, which is why the file records the one it was mined under rather
    than the checker reading it as its own.

    Args:
        path: The model file, from `paths.DECLARE_MODEL`.
    Returns:
        The settings its header records, or `None` for a model written before the header existed,
        which says nothing about how it was mined.
    Raises:
        pydantic.ValidationError: If the header is there but does not describe a discovery.
    """
    for line in path.read_text().splitlines():
        if line.startswith(SETTINGS_LINE):
            return DeclareConfig.model_validate_json(line.removeprefix(SETTINGS_LINE))
    return None


def read_constraints(path: Path) -> list[Constraint]:
    """Read a written declarative model and return the constraints it holds.

    This reads the model directly instead of using `DeclareModel.parse_from_file`, whose grammar
    does not support arbitrary activity names: names containing parentheses or colons can be
    misparsed as declarations or attribute assignments. Constraint lines delimit their activities
    with brackets, so their names can be recovered reliably from the constraint itself.

    Args:
        path: The model file produced by `discover_declare_model`.
    Returns:
        One entry per constraint, in the order the file holds them.
    Raises:
        ValueError: If a line names a template `TEMPLATES` does not hold, if a binary constraint
            does not name two activities, or if it names the same one twice. Each would silently
            change every conformance number in a report, so none is skipped.
    """
    constraints = []

    for raw in path.read_text().splitlines():
        line = raw.strip()

        # Skip the header, which says how the model was mined rather than what it holds, and any
        # other line that is not a constraint.
        if line.startswith(COMMENT) or not _CONSTRAINT_LINE.search(line):
            continue

        head, rest = line.split('[', 1)
        named = _TEMPLATE_AND_CARDINALITY.search(head)
        if named is None:
            raise ValueError(f'"{line}" does not name a template.')

        name, cardinality = named.group(1), named.group(2)
        template = TEMPLATES.get(name)
        if template is None:
            raise ValueError(
                f'"{line}" uses the {name} template, which {__name__} does not check. '
                f'Add it to TEMPLATES, or mine the model without it.'
            )

        activities = rest.split(']')[0].split(', ')
        expected = 2 if template.is_binary else 1
        if len(activities) != expected:
            raise ValueError(f'"{line}" names {len(activities)} activities, not {expected}.')
        if template.is_binary and activities[0] == activities[1]:
            raise ValueError(
                f'"{line}" names one activity twice, which no template here is defined for.'
            )

        constraints.append(
            Constraint(
                template=template,
                first=activities[0],
                second=activities[1] if template.is_binary else None,
                n=int(cardinality) if template.supports_cardinality and cardinality else 1,
            )
        )
    return constraints


class ConformanceChecker:
    """Scores traces against the declarative model a dataset was mined for.

    Reads nothing off disk per check, so a scoring pool builds one per worker and reuses it.
    """

    def __init__(self, dataset: str) -> None:
        """
        Args:
            dataset: The dataset whose model to check against, read from where preprocessing
                wrote it.
        """
        self._constraints = tuple(read_constraints(paths.DECLARE_MODEL.require(dataset)))

    @lru_cache(maxsize=_TRACE_CACHE_SIZE)  # noqa: B019 -- one checker per scoring process
    def rate(self, trace: tuple[str, ...]) -> float:
        """
        The fraction of the model's constraints one trace satisfies.

        Args:
            trace: The trace's activity names, in order. A whole case, prefix included: a
                constraint like `Init` or `Precedence` is about the trace, not about a run of
                events inside it.
        Returns:
            The satisfied share, in `[0, 1]`, or 0.0 for a model that checks nothing.
        """
        positions: Positions = {}
        for index, activity in enumerate(trace):
            positions.setdefault(activity, []).append(index)

        satisfied = sum(constraint.holds(trace, positions) for constraint in self._constraints)
        return satisfied / len(self._constraints) if self._constraints else 0.0
