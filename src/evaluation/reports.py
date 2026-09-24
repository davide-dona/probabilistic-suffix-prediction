import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Self

import pandas as pd
from pydantic import TypeAdapter, ValidationError

from src.evaluation.scoring import EvaluationSummary, LengthSummary
from src.runs.provenance import ArtifactProvenance


@dataclass(frozen=True)
class EvaluationReport:
    """Evaluation results and source artifact provenance.

    metadata identifies the dataset, model, training run, dataset fingerprint, and checkpoint
    hash. summary contains overall scores and both observed-length breakdowns.
    """

    metadata: dict[str, str]
    summary: EvaluationSummary

    @classmethod
    def read(cls, path: str | Path) -> Self:
        """Read and validate a JSON evaluation report.

        Args:
            path: JSON report containing an evaluation summary and artifact provenance.

        Returns:
            The validated report, including its run and checkpoint metadata.

        Raises:
            ValidationError: If the report does not match the expected structure.
            ValueError: If the JSON or artifact provenance is invalid.
        """
        path = Path(path)
        report = _REPORT_ADAPTER.validate_python(json.loads(path.read_bytes()))
        ArtifactProvenance.from_metadata(report.metadata)
        return report

    def write(self, path: str | Path) -> Path:
        """Write the report as JSON.

        Args:
            path: Destination file in an existing directory, replacing any existing contents.

        Returns:
            The destination path.
        """
        path = Path(path)
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(asdict(self), indent=4))
        temporary.replace(path)
        return path


_REPORT_ADAPTER = TypeAdapter(EvaluationReport)


class Axis(StrEnum):
    """Metric aggregation levels."""

    OVERALL = 'overall'
    PREFIX = 'prefix'
    SUFFIX = 'suffix'


REPORT_COLUMNS = ('dataset', 'model', 'axis', 'length', 'prefixes', 'metric', 'value')


def _group_by_model(reports: Iterable[tuple[dict[str, str], Path]]) -> dict[str, dict[str, Path]]:
    """Group one evaluation artifact per model under each dataset, rejecting duplicates."""
    grouped: dict[str, dict[str, Path]] = {}
    for metadata, path in reports:
        dataset, model = metadata['dataset'], metadata['model']
        models = grouped.setdefault(dataset, {})
        if model in models:
            raise ValueError(f'{dataset} has two reports for {model}: {models[model]}, {path}')
        models[model] = path
    return grouped


def _summary_rows(
    summary: EvaluationSummary | LengthSummary,
    *,
    identity: dict[str, str],
    axis: Axis,
    length: int | None,
) -> Iterator[dict[str, object]]:
    for metric, value in summary.scores.flatten().items():
        yield identity | {
            'axis': axis,
            'length': length,
            'prefixes': summary.prefixes,
            'metric': metric,
            'value': value,
        }


def _report_rows(report: EvaluationReport) -> Iterator[dict[str, object]]:
    """Yield overall scores followed by prefix-length and suffix-length breakdowns."""
    identity = {'dataset': report.metadata['dataset'], 'model': report.metadata['model']}
    summary = report.summary
    yield from _summary_rows(summary, identity=identity, axis=Axis.OVERALL, length=None)
    for entry in summary.by_prefix_length:
        yield from _summary_rows(entry, identity=identity, axis=Axis.PREFIX, length=entry.length)
    for entry in summary.by_suffix_length:
        yield from _summary_rows(entry, identity=identity, axis=Axis.SUFFIX, length=entry.length)


def read_reports(files: Sequence[Path]) -> pd.DataFrame:
    """Load reports into the dataframe consumed by tables and figures.

    Args:
        files: JSON evaluation reports, with at most one report per dataset and model.

    Returns:
        Long-form scores with REPORT_COLUMNS, ordered by report, aggregation level, bucket,
        and registered metric. Overall rows have a missing length; empty input retains the schema.

    Raises:
        ValueError: If a report or its provenance is invalid, or a dataset repeats a model.
    """
    reports: list[tuple[Path, EvaluationReport]] = []
    for file in files:
        try:
            reports.append((file, EvaluationReport.read(file)))
        except ValidationError as error:
            raise ValueError(f'{file} is not an evaluation report: {error}') from error
    _group_by_model((report.metadata, file) for file, report in reports)
    rows = [row for _, report in reports for row in _report_rows(report)]
    frame = pd.DataFrame(rows, columns=list(REPORT_COLUMNS))
    return frame.astype({'length': 'Int64', 'prefixes': 'Int64', 'value': 'float64'})
