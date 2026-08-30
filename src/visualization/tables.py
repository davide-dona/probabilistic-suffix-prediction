from collections.abc import Container, Sequence

import pandas as pd

from src.visualization import labels
from src.visualization.catalogue import Column, Table

# What a cell reads where the row has no value for the column: a log-only column on a model's row,
# or a column with no target on the log's own row.
MISSING = '-'
# What the log's own row is called in the Model column.
LOG_ROW = 'Log'


def _escape_latex(text: str) -> str:
    """Escape the characters a dataset name or a model's label could carry into LaTeX text mode."""
    for character in ('\\', '&', '%', '$', '#', '_', '{', '}'):
        text = text.replace(character, f'\\{character}')
    return text


def _value(frame: pd.DataFrame, key: str | None) -> float | None:
    """The one value a metric has in a set of rows, or `None` where it has none.

    Args:
        frame: The rows of one model and one log, from `read_reports`.
        key: Which metric to read, or `None` for a column that has none for this row.
    Returns:
        The metric's value, or `None` where the column names no metric for this row or the run
        never reported one.
    """
    if key is None:
        return None
    rows = frame.loc[frame['metric'] == key, 'value']
    return float(rows.iloc[0]) if len(rows) else None


def _row(
    columns: Sequence[Column],
    frame: pd.DataFrame,
    *,
    label: str,
    metrics: Sequence[str | None],
    best: Container[str],
) -> str:
    """One row of a dataset's block: its name, then a formatted cell per column.

    Args:
        columns: The table's columns, in the order it writes them.
        frame: The rows of the run this table row reports, from `read_reports`.
        label: What the Model column reads, a model's label or `LOG_ROW`.
        metrics: Which metric each column reads for this row, `None` where it has none.
        best: The metrics this row is emphasized on, from `test_significance`. Empty for the log's
            own row, which is a reference rather than a competitor.
    Returns:
        The row as a LaTeX line, its cells `MISSING` wherever there is nothing to write.
    """
    cells = []
    for column, key in zip(columns, metrics, strict=True):
        value = _value(frame, key)
        if value is None:
            cells.append(MISSING)
            continue
        written = column.entry.format(value)
        cells.append(f'\\textbf{{{written}}}' if key in best else written)
    return '   & ' + ' & '.join([_escape_latex(label), *cells]) + ' \\\\'


def _block(
    table: Table,
    frame: pd.DataFrame,
    significance: pd.DataFrame,
    models: Sequence[str],
) -> list[str]:
    """One log's block: the log's own row where the table has one, then a row per model.

    Args:
        table: Which table is being rendered.
        frame: The rows of this log alone, from `read_reports`.
        significance: Which models are tied with the best of each metric on this log.
        models: The models to write, in the order the table writes them. A model with no report
            for this log is left out rather than written as a row of dashes.
    Returns:
        The block's lines, the first of them opening the `\\multirow` that names the log.
    """
    scored = [model for model in models if (frame['model'] == model).any()]
    rows = []

    if table.has_log_row:
        # Every model of a log carries the same value for a log's metric, so any of them says it.
        rows.append(
            _row(
                table.columns,
                frame[frame['model'] == scored[0]],
                label=LOG_ROW,
                metrics=[column.log.key if column.log else None for column in table.columns],
                best=(),
            )
        )

    for model in scored:
        marked = significance[(significance['model'] == model) & significance['best']]
        rows.append(
            _row(
                table.columns,
                frame[frame['model'] == model],
                label=labels.MODELS[model].label,
                metrics=[column.models.key if column.models else None for column in table.columns],
                best=set(marked['metric']),
            )
        )
    return rows


def latex_table(frame: pd.DataFrame, table: Table, significance: pd.DataFrame) -> str:
    """Render one table as booktabs, ready to be input into a paper.

    Args:
        frame: Every report read, from `read_reports`. Only the overall rows are tabulated: a
            table compares runs over their whole split.
        table: Which table to render.
        significance: Which models are the best or tied with it, from `test_significance`, under
            the same names as `frame`.
    Returns:
        The tabular alone, one column per metric and one block of rows per log, the best value of
        each column in bold along with every value the test cannot separate from it. Needs the
        `booktabs`, `multirow` and `tabularx` packages, and belongs inside the paper's own float,
        which is where its caption and label are written. The wider tables are meant for a
        full-width float.
    """
    overall = frame[frame['axis'] == table.axis]
    # One row per model within one block per log, both in the order they are declared, so two
    # tables of the same runs read the same way.
    models = labels.MODELS.ordered(overall['model'])
    datasets = labels.DATASETS.ordered(overall['dataset'])
    headers = [column.entry.table_header for column in table.columns]

    lines = ['\\toprule', '  Dataset & Model & ' + ' & '.join(headers) + ' \\\\', '\\midrule']
    for index, dataset in enumerate(datasets):
        if index > 0:
            lines.append('\\midrule')
        rows = _block(
            table,
            overall[overall['dataset'] == dataset],
            significance[significance['dataset'] == dataset],
            models,
        )
        # `*` rather than `=`: the Dataset column sizes itself to its content, so the label is
        # set at its natural width and the column widens to it.
        name = _escape_latex(labels.DATASETS[dataset])
        lines.append(f'  \\multirow{{{len(rows)}}}{{*}}{{{name}}}')
        lines.extend(rows)
    lines.append('\\bottomrule')

    value_columns = f'*{{{len(table.columns)}}}{{>{{\\centering\\arraybackslash}}X}}'
    preamble = f'\\begin{{tabularx}}{{\\linewidth}}{{ll|{value_columns}}}'
    return '\n'.join((preamble, *lines, '\\end{tabularx}')) + '\n'
