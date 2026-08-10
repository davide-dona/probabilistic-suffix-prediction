from collections.abc import Sequence
from dataclasses import dataclass

from src.visualization.metrics import (
    CONFORMANCE_MEAN,
    CONFORMANCE_POINT,
    DLS_MEAN,
    DLS_POINT,
    ENERGY_SCORE,
    HIT_RATE_AT_1,
    HIT_RATE_AT_5,
    HIT_RATE_AT_10,
    LENGTH_AE_MEAN,
    LENGTH_AE_POINT,
    REMAINING_TIME_AE_MEAN,
    REMAINING_TIME_AE_POINT,
    SAMPLE_DIVERSITY,
    SUFFIX_LENGTH,
    UNIQUE_SAMPLE_RATE,
    MetricSpec,
    format_value,
    metric_value,
)
from src.visualization.runs import PlottedRun

# What a cell reads where a model was never scored on a dataset, or where a metric has no value
# under one estimator.
MISSING = '-'


@dataclass(frozen=True)
class MetricRow:
    """One block of rows: what the block measures, and which metric each estimator column reads.

    A metric read two ways is one row rather than two, so the estimators sit beside each other
    under a model and the reader compares them across the page instead of down it.
    """

    label: str  # What the Metric column reads, without its unit or its direction
    # One entry per estimator of the table, in the same order; `None` where the metric has no
    # value under that estimator.
    metrics: tuple[MetricSpec | None, ...]


@dataclass(frozen=True)
class TableSpec:
    """Specifies one table: what its files are called, its estimator columns and its rows."""

    name: str  # What its files are called
    # The sub-column each model is split into, e.g. the most likely answer beside the sampled one.
    # A single empty label is a table with one column per model and no sub-header.
    estimators: tuple[str, ...]
    rows: tuple[MetricRow, ...]

    def __post_init__(self) -> None:
        for row in self.rows:
            if len(row.metrics) != len(self.estimators):
                raise ValueError(
                    f'row {row.label!r} of table {self.name!r} names {len(row.metrics)} metrics '
                    f'for {len(self.estimators)} estimator columns.'
                )


# The generated suffix decoded from the mean of `p(z | prefix)`, beside the mean over samples
# drawn from it: one answer to commit to, and what the whole predictive distribution is worth.
ESTIMATORS = ('most likely', 'prob.')

QUALITY_TABLE = TableSpec(
    name='comparison-quality',
    estimators=ESTIMATORS,
    rows=(
        MetricRow(label='Suffix DL similarity', metrics=(DLS_POINT, DLS_MEAN)),
        MetricRow(label='Suffix conformance', metrics=(CONFORMANCE_POINT, CONFORMANCE_MEAN)),
        MetricRow(label='Suffix length AE', metrics=(LENGTH_AE_POINT, LENGTH_AE_MEAN)),
        MetricRow(
            label='Remaining time AE',
            metrics=(REMAINING_TIME_AE_POINT, REMAINING_TIME_AE_MEAN),
        ),
    ),
)

# What only a set of samples can say: the metrics with no single-answer counterpart to sit beside,
# which is what keeps them out of `QUALITY_TABLE` rather than any difference in importance.
SAMPLED_METRICS = (
    HIT_RATE_AT_1,
    HIT_RATE_AT_5,
    HIT_RATE_AT_10,
    ENERGY_SCORE,
    SAMPLE_DIVERSITY,
    UNIQUE_SAMPLE_RATE,
    SUFFIX_LENGTH,
)

SAMPLED_TABLE = TableSpec(
    name='comparison-samples',
    estimators=('',),
    rows=tuple(MetricRow(label=spec.label, metrics=(spec,)) for spec in SAMPLED_METRICS),
)

TABLES = (QUALITY_TABLE, SAMPLED_TABLE)


def _models(grouped: dict[str, list[PlottedRun]]) -> list[str]:
    """The labels of every model tabulated, in the order they were first named.

    Args:
        grouped: The runs of each dataset, from `load_runs`.
    Returns:
        One entry per label, since a model scored on two datasets is one column group.
    """
    return list(dict.fromkeys(run.label for runs in grouped.values() for run in runs))


def _row_header(row: MetricRow, *, up: str, down: str) -> str:
    """One block's Metric cell: what it measures, its unit and which way is better.

    Args:
        row: The block to name. Its metrics share a unit and a direction, so the first of them
            answers for all.
        up: How to write the arrow of a metric to maximize, which is per output format.
        down: How to write the arrow of a metric to minimize.
    Returns:
        The label, its unit appended where it has one and its arrow where it has a best value.
    """
    spec = next(metric for metric in row.metrics if metric is not None)
    unit = '' if spec.unit is None else f' [{spec.unit}]'
    direction = '' if spec.higher_is_better is None else f' {up if spec.higher_is_better else down}'
    return f'{row.label}{unit}{direction}'


def _cells(row: MetricRow, runs: dict[str, PlottedRun], models: Sequence[str]) -> list[str | None]:
    """The values of one row, one per model and estimator, `None` where there is nothing to show.

    Args:
        row: The block the row belongs to, naming the metric of each estimator column.
        runs: The runs scored on this row's dataset, keyed by label.
        models: Every model tabulated, which is the column groups in order.
    Returns:
        One entry per column, the models in order and the estimators within each, formatted where
        the model was scored on this dataset and the metric has a value under that estimator.
    """
    values: list[str | None] = []
    for model in models:
        run = runs.get(model)
        for spec in row.metrics:
            if run is None or spec is None:
                values.append(None)
            else:
                values.append(format_value(metric_value(run.report.metrics.scores, spec), spec))
    return values


def _best(row: MetricRow, cells: Sequence[str | None]) -> set[int]:
    """The columns of one row holding its best value.

    Args:
        row: The block the row belongs to, whose metrics say which way is better.
        cells: The row's formatted values, from `_cells`.
    Returns:
        The indices of the best cells, or none of them where the metric has no best value or where
        there is only one value to be best. Compared as formatted, so two cells printed alike are
        both best rather than one winning on a decimal the table does not show.
    """
    spec = next(metric for metric in row.metrics if metric is not None)
    present = {index: cell for index, cell in enumerate(cells) if cell is not None}
    if spec.higher_is_better is None or len(present) < 2:
        return set()
    best = (max if spec.higher_is_better else min)(present.values(), key=float)
    return {index for index, cell in present.items() if cell == best}


def _blocks(
    grouped: dict[str, list[PlottedRun]], spec: TableSpec, models: Sequence[str], emphasis: str
) -> list[list[tuple[str, list[str]]]]:
    """The body of one table, one block of rows per metric.

    Args:
        grouped: The runs of each dataset, from `load_runs`, keyed by what the Dataset column
            should read rather than by the log's own name.
        spec: The table to build, naming its rows and estimator columns.
        models: Every model tabulated, which is the column groups in order.
        emphasis: A format string with one `{}`, wrapped around the best value of a row.
    Returns:
        One block per metric, each a list of `(dataset, cells)` rows in the order the datasets were
        named, each cell already formatted and emphasized.
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
                        else (emphasis.format(cell) if index in best else cell)
                        for index, cell in enumerate(cells)
                    ],
                )
            )
        blocks.append(rows)
    return blocks


def _markdown_row(cells: Sequence[str], widths: Sequence[int]) -> str:
    """One markdown row, its cells padded so the raw text lines up into columns."""
    padded = (cell.ljust(width) for cell, width in zip(cells, widths, strict=True))
    return f'| {" | ".join(padded)} |'


def markdown_table(grouped: dict[str, list[PlottedRun]], spec: TableSpec) -> str:
    """Render one table as markdown, the form the console preview is printed in.

    Args:
        grouped: The runs of each dataset, keyed as `_blocks` takes them.
        spec: The table to render.
    Returns:
        The table alone, with the best value of each row in bold and every column padded so the
        raw text reads as a table too. Markdown cannot span a header across a group of columns,
        so each column is headed by its model and estimator together and each row repeats its
        metric.
    """
    models = _models(grouped)
    headers = [
        'Metric',
        'Dataset',
        *(
            model if not estimator else f'{model} ({estimator})'
            for model in models
            for estimator in spec.estimators
        ),
    ]
    rows = [
        [_row_header(row, up='↑', down='↓'), dataset, *cells]
        for row, block in zip(spec.rows, _blocks(grouped, spec, models, '**{}**'), strict=True)
        for dataset, cells in block
    ]
    widths = [
        max(len(cell) for cell in (header, *(row[index] for row in rows)))
        for index, header in enumerate(headers)
    ]
    return (
        '\n'.join(
            (
                _markdown_row(headers, widths),
                '|' + '|'.join('-' * (width + 2) for width in widths) + '|',
                *(_markdown_row(row, widths) for row in rows),
            )
        )
        + '\n'
    )


def _escape_latex(text: str) -> str:
    """Escape the characters a dataset name or a run's label could carry into LaTeX text mode."""
    for character in ('\\', '&', '%', '$', '#', '_', '{', '}'):
        text = text.replace(character, f'\\{character}')
    return text


def _latex_columns(spec: TableSpec, models: Sequence[str]) -> str:
    """The column spec of one table: the two label columns, then one right-aligned column per
    model and estimator.

    Args:
        spec: The table to lay out, whose estimators say how wide a model's group is.
        models: Every model tabulated, which is the column groups in order.
    Returns:
        The argument of `tabular`, with the groups set apart by a gap so the estimators of one
        model read as belonging together rather than as a run of numbers.
    """
    group = 'r' * len(spec.estimators)
    return 'll|' + '@{\\hspace{2em}}'.join([group] * len(models))


def _latex_header(spec: TableSpec, models: Sequence[str]) -> list[str]:
    """The header lines of one table: the models, the rules grouping them and their estimators.

    Args:
        spec: The table to head, naming its estimator columns.
        models: Every model tabulated, which is the column groups in order.
    Returns:
        The lines above the first `\\midrule`. A table whose estimators are unlabelled gets a
        single header row, since there is nothing to group.
    """
    width = len(spec.estimators)
    names = [_escape_latex(model) for model in models]
    if width == 1 and not spec.estimators[0]:
        return [f'  Metric & Dataset & {" & ".join(names)} \\\\']

    # Column 1 is Metric and column 2 Dataset, so the first model starts at column 3.
    starts = [3 + index * width for index in range(len(models))]
    return [
        '  Metric & Dataset & '
        + ' & '.join(f'\\multicolumn{{{width}}}{{c}}{{{name}}}' for name in names)
        + ' \\\\',
        ''.join(f'\\cmidrule(lr){{{start}-{start + width - 1}}}' for start in starts),
        '   & & ' + ' & '.join(spec.estimators * len(models)) + ' \\\\',
    ]


def latex_table(grouped: dict[str, list[PlottedRun]], spec: TableSpec) -> str:
    """Render one table as booktabs, ready to be input into a paper.

    Args:
        grouped: The runs of each dataset, keyed as `_blocks` takes them.
        spec: The table to render.
    Returns:
        The tabular alone, with each model spanning its estimator columns, the best value of
        each row in bold and a rule between metrics. Needs the `booktabs`, `multirow` and
        `graphicx` packages, and belongs inside the paper's own float, which is where its
        caption and label are written.
    """
    models = _models(grouped)

    # Only the dataset and the model are free text; the row headers are written with LaTeX arrows
    # and the values are formatted numbers already wrapped in the emphasis this asked for.
    blocks = []
    for row, block in zip(spec.rows, _blocks(grouped, spec, models, '\\textbf{{{}}}'), strict=True):
        label = _row_header(row, up=r'$\uparrow$', down=r'$\downarrow$')
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
                f'\\begin{{tabular}}{{{_latex_columns(spec, models)}}}',
                '\\toprule',
                *_latex_header(spec, models),
                '\\midrule',
                ' \\\\\n\\midrule\n'.join(blocks) + ' \\\\',
                '\\bottomrule',
                '\\end{tabular}%',
                '}',
            )
        )
        + '\n'
    )
