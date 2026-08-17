import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path
from types import CodeType
from typing import Any

import pandas as pd
import pm4py
from Declare4Py.D4PyEventLog import D4PyEventLog
from Declare4Py.ProcessMiningTasks.Discovery.DeclareMiner import DeclareMiner
from Declare4Py.ProcessModels.DeclareModel import (
    DeclareModelConditionParserUtility,
    DeclareModelTemplate,
)
from Declare4Py.Utils.Declare.Checkers import CheckerResult, TemplateConstraintChecker
from Declare4Py.Utils.Declare.TraceStates import TraceState

from src import paths
from src.configs import DeclareConfig
from src.logs.keys import ACTIVITY_KEY, CASE_KEY, TIMESTAMP_KEY

# Parse the declarative model written by `discover_declare_model` back into the constraints it
# holds. Read here rather than through `DeclareModel.parse_from_file`, whose grammar rejects the
# activity names a real log carries: see `_read_constraints`.
_CONSTRAINT_LINE = re.compile(r'^(.*)\[(.*)\]\s*(.*)$')
_TEMPLATE_AND_CARDINALITY = re.compile(r'(^.+?)(\d*$)')

# Maximum number of distinct traces cached by the ConformanceChecker.
# Each prefix generates one trace per sample, and the same trace may be
# generated for multiple prefixes. Caching previously computed scores avoids
# redundant conformance checks and can significantly reduce computation.
_TRACE_CACHE_SIZE = 100_000


def discover_declare_model(
    train: pd.DataFrame,
    *,
    dataset: str,
    declare_config: DeclareConfig,
) -> int:
    """
    Discover a declarative model from the train split and write it beside the dataset.

    Args:
        train: The train split, as preprocessing holds it. The only log discovery reads, so the
            constraints never carry anything from the validation or test split.
        dataset: The dataset the split came from, naming where the model goes.
        declare_config: The `declare` section: which constraints are looked for and how much of
            the log has to support one.

    Returns:
        The number of constraints written.
    """
    # Drop any columns that are not the structural ones
    event_log = pm4py.convert_to_event_log(train[[CASE_KEY, ACTIVITY_KEY, TIMESTAMP_KEY]])
    # Converting from a DataFrame leaves these unset, and D4PyEventLog reads them.
    event_log.properties['pm4py:param:activity_key'] = ACTIVITY_KEY
    event_log.properties['pm4py:param:timestamp_key'] = TIMESTAMP_KEY

    miner = DeclareMiner(
        log=D4PyEventLog(case_name=CASE_KEY, log=event_log),
        consider_vacuity=declare_config.consider_vacuity,
        min_support=declare_config.min_support,
        itemsets_support=declare_config.itemsets_support,
        max_declare_cardinality=declare_config.max_cardinality,
    )
    model = miner.run()

    # A constraint line names its activities inside brackets, separated by `, `, and closes on
    # the conditions behind a `|`. An activity carrying any of that would be read back as two
    # activities, or as a condition, so the model could not say what it was mined to say.
    unwritable = [
        activity
        for activity in model.activities
        if any(marker in activity for marker in ('[', ']', '|', ', '))
    ]
    if unwritable:
        raise ValueError(
            f'{dataset} has activities a declarative model cannot name: {unwritable}. '
            'Rename them in the raw log, or drop them from the preprocessed split.'
        )

    # Write the activities in the Declare4py format
    lines = [f'activity {activity}' for activity in model.activities]
    # Write the constraints in the Declare4py format, one per line
    for constraint, serialized in zip(model.constraints, model.serialized_constraints, strict=True):
        # Declare4Py serializes a binary constraint's two conditions as the one empty field a
        # unary constraint gets, which its own parser then rejects; the missing separator is
        # added back here.
        lines.append(f'{serialized} |' if constraint['template'].is_binary else serialized)

    path = paths.declare_model_path(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n')

    return len(model.constraints)


def _read_constraints(path: Path) -> list[tuple[str, dict[str, Any]]]:
    """Read a written declarative model and return its constraints.

    This reads the model directly instead of using `DeclareModel.parse_from_file`,
    whose grammar does not support arbitrary activity names. For example, names
    containing parentheses or colons can be misparsed as declarations or attribute
    assignments. Constraint lines, however, delimit their activities with brackets,
    so their names can be recovered reliably from the constraint itself.

    Args:
        path: The model file produced by `discover_declare_model`.

    Returns:
        One entry per supported constraint, containing the template and the
        activities, conditions, and cardinality expected by
        `TemplateConstraintChecker`. Constraints using templates unknown to
        Declare4Py are skipped, matching the behaviour of its parser.
    """
    constraints = []

    for line in path.read_text().splitlines():
        line = line.strip()

        # Skip lines that are not constraints
        if not _CONSTRAINT_LINE.search(line):
            continue

        head, activities = line.split('[', 1)
        # Get the template and cardinality from the head of the line
        named = _TEMPLATE_AND_CARDINALITY.search(head)
        if named is None:
            continue

        template = DeclareModelTemplate.get_template_from_string(named.group(1))
        if template is None:
            continue
        cardinality = named.group(2)
        # Build the constraint dictionary expected by TemplateConstraintChecker
        constraint: dict[str, Any] = {
            'template': template,
            'activities': activities.split(']')[0].split(', '),
            # Everything past the activities, one condition per `|`-separated field.
            'condition': re.split(r'\s+\|', line)[1:],
        }
        if template.supports_cardinality:
            constraint['n'] = int(cardinality) if cardinality else 1
        constraints.append((line, constraint))
    return constraints


class _CompilingConditionParser(DeclareModelConditionParserUtility):
    """Compile Declare conditions once and reuse the resulting code objects.

    Declare4Py's template checkers pass conditions directly to `eval`. While
    `eval` accepts both strings and code objects, string inputs are parsed and
    compiled on every call. A mined model typically reuses the same small set of
    conditions across many constraints, and each condition may be evaluated for
    every matching event in every trace. Caching compiled conditions therefore
    replaces repeated parsing with a dictionary lookup.
    """

    @cache
    def parse_data_cond(self, condition: str) -> CodeType:
        return super().parse_data_cond(condition)

    @cache
    def parse_time_cond(self, condition: str) -> CodeType:
        return super().parse_time_cond(condition)


@dataclass(frozen=True)
class _PreparedConstraint:
    """One constraint of a model, resolved down to what checking a trace against it needs.

    The activities and the conditions a constraint is checked with are the same for every trace,
    so they are read off the model once rather than rebuilt per check.
    """

    # The bound template checker to call, already tied to the trace holder it reads from.
    check: Callable[[], CheckerResult]
    activities: list[str]
    rules: dict[str, Any]


class ConformanceChecker:
    """Scores traces against the declarative model a dataset was mined for.

    Holds one trace at a time and swaps the constraint under it, so a check costs the constraint
    and nothing around it. Not shareable between threads or processes for that reason: a scoring
    pool builds one per worker.
    """

    def __init__(self, dataset: str, *, consider_vacuity: bool) -> None:
        """
        Args:
            dataset: The dataset whose model to check against, read from where preprocessing
                wrote it.
            consider_vacuity: Whether a constraint a trace never activates counts as satisfied.
                False makes it count as violated instead, so the rate only credits constraints
                the trace actually exercises.
        """
        # The checkers only ever read `event[ACTIVITY_KEY]` and the trace's length, so a trace is
        # a list of one-key dicts: no event log, and nothing read off disk per check. Every trace
        # is judged as a finished case, which the checkers assume anyway.
        self._trace_holder = TemplateConstraintChecker(
            traces=[],
            completed=True,
            activities=[],
            rules={},
            concept_name=ACTIVITY_KEY,
        )
        self._trace_holder.declare_parser_utility = _CompilingConditionParser()
        self._constraints = self._prepare(
            constraints=_read_constraints(paths.declare_model_path(dataset)),
            consider_vacuity=consider_vacuity,
        )

    def _prepare(
        self,
        *,
        constraints: list[tuple[str, dict[str, Any]]],
        consider_vacuity: bool,
    ) -> list[_PreparedConstraint]:
        """Resolve a model's constraints into what each check needs.

        Args:
            constraints: The model's constraints beside the line each was read from, from
                `_read_constraints`.
            consider_vacuity: Whether a constraint a trace never activates counts as satisfied.
        Returns:
            One entry per constraint whose conditions compile. A constraint whose conditions do
            not is reported and dropped, so it counts towards neither side of the rate.
        """
        parser = self._trace_holder.declare_parser_utility
        prepared = []
        for line, constraint in constraints:
            template = constraint['template']
            rules: dict[str, Any] = {
                'vacuous_satisfaction': consider_vacuity,
                'activation': constraint['condition'][0],
                # The time condition is always at the last position.
                'time': constraint['condition'][-1],
            }
            if template.supports_cardinality:
                rules['n'] = constraint['n']
            if template.is_binary:
                rules['correlation'] = constraint['condition'][1]

            checker_name = f'mp{template.templ_str.replace(" ", "")}'
            check = getattr(self._trace_holder, checker_name, None)
            if check is None:
                raise NotImplementedError(
                    f'Declare4Py has no checker for the {template.templ_str} template.'
                )
            try:
                # Compiled here so a malformed condition is reported once rather than per trace.
                parser.parse_data_cond(rules['activation'])
                parser.parse_time_cond(rules['time'])
                if template.is_binary:
                    parser.parse_data_cond(rules['correlation'])
            except SyntaxError:
                print(f'Condition not properly formatted for constraint "{line}".')
                continue
            prepared.append(
                _PreparedConstraint(
                    check=check,
                    activities=constraint['activities'],
                    rules=rules,
                )
            )
        return prepared

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
        self._trace_holder.traces = [{ACTIVITY_KEY: activity} for activity in trace]
        satisfied = 0
        for constraint in self._constraints:
            self._trace_holder.activities = constraint.activities
            self._trace_holder.rules = constraint.rules
            satisfied += constraint.check().state == TraceState.SATISFIED
        return satisfied / len(self._constraints) if self._constraints else 0.0
