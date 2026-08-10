from collections.abc import Sequence
from dataclasses import dataclass

from src.visualization.metrics import (
    CONFORMANCE_MEAN,
    CONFORMANCE_POINT,
    DLS_BEST,
    DLS_MEAN,
    DLS_POINT,
    ENERGY_SCORE,
    HIT_RATE_AT_1,
    HIT_RATE_AT_5,
    HIT_RATE_AT_10,
    LENGTH_AE_POINT,
    REMAINING_TIME_AE_POINT,
    MetricSpec,
    format_value,
    metric_value,
)
from src.visualization.runs import PlottedRun

# What a cell reads where a model was never scored on a dataset.
MISSING = '-'


@dataclass(frozen=True)
class MetricRow:
    """One block of rows: which metric it reads, and what the Metric column calls it."""

    label: str  # What the Metric column reads, without its unit or its direction
    metric: MetricSpec


@dataclass(frozen=True)
class TableSpec:
    """Specifies one table: what its files are called and which metrics it puts in rows."""

    name: str  # What its files are called
    rows: tuple[MetricRow, ...]


# The two tables answer two questions, and each of them holds a single estimator so that the
# emphasis compares models rather than a model against itself. A point estimate beats the mean of
# ten stochastic draws almost by construction, so the two must never share a row.
POINT_TABLE = TableSpec(
    name='comparison-point',
    rows=(
        MetricRow(label='Suffix DL similarity', metric=DLS_POINT),
        MetricRow(label='Suffix conformance', metric=CONFORMANCE_POINT),
        MetricRow(label='Suffix length AE', metric=LENGTH_AE_POINT),
        MetricRow(label='Remaining time AE', metric=REMAINING_TIME_AE_POINT),
    ),
)

PROBABILISTIC_TABLE = TableSpec(
    name='comparison-probabilistic',
    rows=(
        MetricRow(label='Energy score', metric=ENERGY_SCORE),
        MetricRow(label='Hit rate @1', metric=HIT_RATE_AT_1),
        MetricRow(label='Hit rate @5', metric=HIT_RATE_AT_5),
        MetricRow(label='Hit rate @10', metric=HIT_RATE_AT_10),
        MetricRow(label='Best-of-k DL similarity', metric=DLS_BEST),
        MetricRow(label='Mean DL similarity', metric=DLS_MEAN),
        MetricRow(label='Mean conformance', metric=CONFORMANCE_MEAN),
    ),
)

TABLES = (POINT_TABLE, PROBABILISTIC_TABLE)


def _models(grouped: dict[str, list[PlottedRun]]) -> list[str]:
    """The labels of every model tabulated, in the order they were first named.

    Args:
        grouped: The runs of each dataset, from `load_runs`.
    Returns:
        One entry per label, since a model scored on two datasets is one column.
    """
    return list(dict.fromkeys(run.label for runs in grouped.values() for run in runs))


def _row_header(row: MetricRow) -> str:
    """One block's Metric cell: what it measures, its unit and which way is better.

    Args:
        row: The block to name.
    Returns:
        The label, its unit appended where it has one and its arrow where it has a best value.
    """
    spec = row.metric
    unit = '' if spec.unit is None else f' [{spec.unit}]'
    arrow = r'$\uparrow$' if spec.higher_is_better else r'$\downarrow$'
    direction = '' if spec.higher_is_better is None else f' {arrow}'
    return f'{row.label}{unit}{direction}'


def _cells(row: MetricRow, runs: dict[str, PlottedRun], models: Sequence[str]) -> list[str | None]:
    """The values of one row, one per model, `None` where there is nothing to show.

    Args:
        row: The block the row belongs to, naming the metric to read.
        runs: The runs scored on this row's dataset, keyed by label.
        models: Every model tabulated, which is the columns in order.
    Returns:
        One entry per column, formatted where the model was scored on this dataset.
    """
    return [
        None
        if (run := runs.get(model)) is None
        else format_value(metric_value(run.report.metrics.scores, row.metric), row.metric)
        for model in models
    ]


def _best(row: MetricRow, cells: Sequence[str | None]) -> set[int]:
    """The columns of one row holding its best value.

    Args:
        row: The block the row belongs to, whose metric says which way is better.
        cells: The row's formatted values, from `_cells`.
    Returns:
        The indices of the best cells, or none of them where the metric has no best value or where
        there is only one value to be best. Compared as formatted, so two cells printed alike are
        both best rather than one winning on a decimal the table does not show.
    """
    spec = row.metric
    present = {index: cell for index, cell in enumerate(cells) if cell is not None}
    if spec.higher_is_better is None or len(present) < 2:
        return set()
    best = (max if spec.higher_is_better else min)(present.values(), key=float)
    return {index for index, cell in present.items() if cell == best}


def _blocks(
    grouped: dict[str, list[PlottedRun]], spec: TableSpec, models: Sequence[str]
) -> list[list[tuple[str, list[str]]]]:
    """The body of one table, one block of rows per metric.

    Args:
        grouped: The runs of each dataset, from `load_runs`, keyed by what the Dataset column
            should read rather than by the log's own name.
        spec: The table to build, naming its rows.
        models: Every model tabulated, which is the columns in order.
    Returns:
        One block per metric, each a list of `(dataset, cells)` rows in the order the datasets were
        named, each cell already formatted and the best of each row in bold.
    """
    by_dataset = {dataset: {run.label: run for run in runs} for dataset, runs in grouped.items()}
    blocks = []
    for row in spec.rows:
        rows = []
        for dataset, runs in by_dataset.items():
            cells = _cells(row, runs, models)
            best = _best(row, cells)
            rows.append(
                (
                    dataset,
                    [
                        MISSING
                        if cell is None
                        else (f'\\textbf{{{cell}}}' if index in best else cell)
                        for index, cell in enumerate(cells)
                    ],
                )
            )
        blocks.append(rows)
    return blocks


def _escape_latex(text: str) -> str:
    """Escape the characters a dataset name or a run's label could carry into LaTeX text mode."""
    for character in ('\\', '&', '%', '$', '#', '_', '{', '}'):
        text = text.replace(character, f'\\{character}')
    return text


def latex_table(grouped: dict[str, list[PlottedRun]], spec: TableSpec) -> str:
    """Render one table as booktabs, ready to be input into a paper.

    Args:
        grouped: The runs of each dataset, keyed as `_blocks` takes them.
        spec: The table to render.
    Returns:
        The tabular alone, with one column per model, the best value of each row in bold and a
        rule between metrics. Needs the `booktabs`, `multirow` and `graphicx` packages, and
        belongs inside the paper's own float, which is where its caption and label are written.
    """
    models = _models(grouped)

    # Only the dataset and the model are free text; the row headers are written with LaTeX arrows
    # and the values are formatted numbers already wrapped in their emphasis.
    blocks = []
    for row, block in zip(spec.rows, _blocks(grouped, spec, models), strict=True):
        label = _row_header(row)
        header = f'\\multirow{{{len(block)}}}{{*}}{{{label}}}'
        blocks.append(
            ' \\\\\n'.join(
                '  ' + ' & '.join([header if index == 0 else '', _escape_latex(dataset), *cells])
                for index, (dataset, cells) in enumerate(block)
            )
        )

    return (
        '\n'.join(
            (
                # `\linewidth` is the column in a `table` and the page in a `table*`, so the table
                # fits whichever the paper puts it in.
                '\\resizebox{\\linewidth}{!}{%',
                f'\\begin{{tabular}}{{ll|{"r" * len(models)}}}',
                '\\toprule',
                '  Metric & Dataset & '
                + ' & '.join(_escape_latex(model) for model in models)
                + ' \\\\',
                '\\midrule',
                ' \\\\\n\\midrule\n'.join(blocks) + ' \\\\',
                '\\bottomrule',
                '\\end{tabular}%',
                '}',
            )
        )
        + '\n'
    )
