from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from src.evaluation.report import EvaluationReport
from src.visualization.style import SeriesStyle, series_styles


@dataclass(frozen=True)
class PlottedRun:
    """One evaluation report, under the name and the look it is drawn and tabulated with."""

    label: str
    style: SeriesStyle
    report: EvaluationReport

    @property
    def dataset(self) -> str:
        """The log the run was scored on, which is the figures it appears in."""
        return self.report.run.dataset


def _read_report(file: Path) -> EvaluationReport:
    """Read one evaluation report, saying which file was not one.

    Args:
        file: The report to read.
    Returns:
        The report it holds.
    Raises:
        ValueError: If the file is not an evaluation report, naming it, since a swept directory
            reads files nobody typed and the schema error names none of them.
    """
    try:
        return EvaluationReport.read(file)
    except ValidationError as error:
        raise ValueError(f'{file} is not an evaluation report: {error}') from error


def load_runs(files: Sequence[Path], labels: Mapping[str, str]) -> dict[str, list[PlottedRun]]:
    """Read a set of evaluation reports and group them by the log they were scored on.

    A label is a model rather than a run, so one model shown on two logs is drawn in the same
    colour throughout.

    Args:
        files: The reports to read, from `pipelines.evaluate`.
        labels: What to call a model in legends and tables, keyed by its own name. A model not
            named here keeps that name.
    Returns:
        The runs of each log, both the logs and the runs within one in the order the files were
        named, so a legend reads in the order they were asked for.
    Raises:
        FileNotFoundError: If a report does not exist.
        ValueError: If a file is not a report, if a label names no report's model, if a log ends
            up with two runs of the same name, or if there are more models than the palette can
            tell apart.
    """
    missing = [file for file in files if not file.exists()]
    if missing:
        raise FileNotFoundError(
            f'no evaluation report at {", ".join(str(file) for file in missing)}. '
            'Run `python -m pipelines.evaluate` first, or name the right report.'
        )

    reports = [_read_report(file) for file in files]

    unmatched = sorted(set(labels) - {report.run.model for report in reports})
    if unmatched:
        raise ValueError(
            f'nothing to label called {", ".join(repr(model) for model in unmatched)}. '
            f'The models read are {", ".join(sorted({report.run.model for report in reports}))}.'
        )
    names = [labels.get(report.run.model, report.run.model) for report in reports]

    # One style per model, assigned in the order the models first appear, so the same model keeps
    # its colour across every figure of every log.
    models = list(dict.fromkeys(names))
    styles = dict(zip(models, series_styles(len(models)), strict=True))

    grouped: dict[str, list[PlottedRun]] = {}
    drawn: dict[tuple[str, str], Path] = {}
    for name, report, file in zip(names, reports, files, strict=True):
        dataset = report.run.dataset
        already = drawn.get((dataset, name))
        if already is not None:
            raise ValueError(
                f'two reports of {dataset} are both called {name!r}:\n  {already}\n  {file}\n'
                'Remove the stale one, or name the reports you want with -e.'
            )
        drawn[dataset, name] = file
        grouped.setdefault(dataset, []).append(
            PlottedRun(label=name, style=styles[name], report=report)
        )
    return grouped
