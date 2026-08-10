from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

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
        return self.report.run.dataset.name


def load_runs(files: Sequence[Path], labels: Sequence[str] | None) -> dict[str, list[PlottedRun]]:
    """Read a set of evaluation reports and group them by the log they were scored on.

    Every variant of a log shares its figures, since they describe the same process. They are not
    the same population, though: two variants cut their test cases differently, so the default
    label names the variant beside the model rather than leaving a series to be read as if every
    curve on the axes had been scored over the same prefixes.

    A label is a model rather than a run, so two reports sharing one are the same model on two
    logs and are drawn in the same colour throughout.

    Args:
        files: The reports to read, from `pipelines.evaluate`.
        labels: What to call each of them, in the same order, or `None` for `<model> (<variant>)`.
    Returns:
        The runs of each log, both the logs and the runs within one in the order the files were
        named, so a legend reads in the order they were asked for.
    Raises:
        FileNotFoundError: If a report does not exist.
        ValueError: If the labels do not match the files one for one, if a log ends up with two
            runs of the same name, or if there are more models than the palette can tell apart.
    """
    if labels is not None and len(labels) != len(files):
        raise ValueError(
            f'got {len(labels)} labels for {len(files)} evaluation reports. Name one label per '
            'report, in the same order, or none at all.'
        )

    missing = [file for file in files if not file.exists()]
    if missing:
        raise FileNotFoundError(
            f'no evaluation report at {", ".join(str(file) for file in missing)}. '
            'Run `python -m pipelines.evaluate` first, or name the right report.'
        )

    reports = [EvaluationReport.read(file) for file in files]
    names = (
        list(labels)
        if labels is not None
        else [f'{report.run.model} ({report.run.dataset.variant})' for report in reports]
    )

    # One style per model, assigned in the order the models first appear, so the same model keeps
    # its colour across every figure of every log.
    models = list(dict.fromkeys(names))
    styles = dict(zip(models, series_styles(len(models)), strict=True))

    grouped: dict[str, list[PlottedRun]] = {}
    for name, report in zip(names, reports, strict=True):
        dataset = report.run.dataset.name
        runs = grouped.setdefault(dataset, [])
        if any(run.label == name for run in runs):
            raise ValueError(
                f'two reports of {dataset} are both called {name!r}. Give them separate '
                'labels, since a figure cannot draw them apart.'
            )
        runs.append(PlottedRun(label=name, style=styles[name], report=report))
    return grouped
