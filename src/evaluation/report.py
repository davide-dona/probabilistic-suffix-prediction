import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Self

import pandas as pd
from pydantic import TypeAdapter, ValidationError

from src.evaluation.metrics import EvaluationMetrics, PairScores
from src.identity import RunIdentity, group_by_model


@dataclass(frozen=True)
class EvaluationReport:
    """Everything one evaluation produced, under the identity of the run it scored."""

    run: RunIdentity
    metrics: EvaluationMetrics

    @classmethod
    def read(cls, path: str | Path) -> Self:
        """Read a report back from the JSON `write` produced.

        Args:
            path: The report to read, e.g. from `paths.evaluation_path`.
        Returns:
            The report, validated against the current schema so a report written before a metric
            was added or renamed fails here rather than on a missing key further down.
        """
        return _ADAPTER.validate_json(Path(path).read_bytes())

    def write(self, path: str | Path) -> Path:
        """
        Write the report as JSON, creating parent directories.

        Args:
            path: Where to write, e.g. the generations file's own path with a `.json` suffix.
        Returns:
            The path written to.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=4))
        return path


# Built once, since a `TypeAdapter` compiles the schema it validates against.
_ADAPTER = TypeAdapter(EvaluationReport)


class Axis(StrEnum):
    """The three breakdowns a metric is read against.

    A figure names the one it draws, and it is what the `axis` column of `read_reports` holds:
    the evaluation as a whole, or one length at a time, cut either at the prefix or at the
    ground-truth suffix.
    """

    OVERALL = 'overall'
    PREFIX = 'prefix'
    SUFFIX = 'suffix'


# What every figure and every table reads. One row is one metric of one run, so a metric added to
# the scores reaches the figures without a change here.
REPORT_COLUMNS = ('dataset', 'model', 'axis', 'length', 'pairs', 'metric', 'value')


def _flatten(scores: PairScores) -> dict[str, float]:
    """Every metric one set of scores holds, keyed by its own name.

    Args:
        scores: An evaluation's means, or the means of the prefixes sharing one length.
    Returns:
        Both families in one namespace, since a metric's name is unique across them and a figure
        asks for a metric rather than for a family.
    """
    return asdict(scores.accuracy) | asdict(scores.conformance)


def _rows(report: EvaluationReport) -> list[dict[str, object]]:
    """Lay one report out as one row per metric per breakdown."""
    run, metrics = report.run, report.metrics
    identity = {'dataset': run.dataset, 'model': run.model}

    # The energy distance is measured over the split as a whole, so it joins the overall means
    # rather than having a value at any one length.
    overall = _flatten(metrics.scores) | {'energy_distance': metrics.energy_distance}
    rows: list[dict[str, object]] = [
        identity
        | {'axis': Axis.OVERALL, 'length': None, 'pairs': metrics.pairs, 'metric': m, 'value': v}
        for m, v in overall.items()
    ]
    breakdowns = ((Axis.PREFIX, metrics.by_prefix_length), (Axis.SUFFIX, metrics.by_suffix_length))
    rows.extend(
        identity
        | {
            'axis': axis,
            'length': entry.length,
            'pairs': entry.pairs_count,
            'metric': metric,
            'value': value,
        }
        for axis, breakdown in breakdowns
        for entry in breakdown
        for metric, value in _flatten(entry.scores).items()
    )
    return rows


def read_reports(files: Sequence[Path]) -> pd.DataFrame:
    """Read a set of evaluation reports into the frame every figure and table is drawn from.

    Args:
        files: The reports to compare, from `pipelines.evaluate`. Each says which run wrote it, so
            they may come from any number of logs and models.
    Returns:
        One row per metric per breakdown, under `REPORT_COLUMNS`. `length` and `pairs` are nullable
        integers, `length` being null on the `Axis.OVERALL` rows.
    Raises:
        ValueError: If a file is not an evaluation report, or if one log is given two runs of the
            same model, which would draw two lines under one name.
    """
    reports: list[tuple[Path, EvaluationReport]] = []
    for file in files:
        try:
            reports.append((file, EvaluationReport.read(file)))
        except ValidationError as error:
            # A swept directory reads files nobody typed, and the schema error names none of them.
            raise ValueError(f'{file} is not an evaluation report: {error}') from error

    # Called for the check alone: a log given two runs of one model would draw two lines under one
    # name. The rows below are laid out per report either way, so the grouping itself is not needed.
    group_by_model((report.run, file) for file, report in reports)

    rows = [row for _, report in reports for row in _rows(report)]
    frame = pd.DataFrame(rows, columns=list(REPORT_COLUMNS))
    return frame.astype({'length': 'Int64', 'pairs': 'Int64', 'value': 'float64'})
