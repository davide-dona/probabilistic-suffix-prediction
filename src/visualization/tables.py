from collections.abc import Mapping, Sequence

import pandas as pd

from src.visualization import labels
from src.visualization.catalogue import MetricEntry, Table

# What a cell reads where a model was never scored on a dataset.
MISSING = '-'


def _cells(entry: MetricEntry, values: Mapping[str, float], models: Sequence[str]) -> list[str]:
    """One row of a block: a formatted cell per model, the best in bold and the second underlined.

    Args:
        entry: Which metric the row reports, naming what counts as better.
        values: What each model scored on it, for the models that were scored at all.
        models: The columns to fill, in the order the table writes them.
    Returns:
        One cell per model, `MISSING` where it was never scored.
    """
    formatted = {model: entry.format(value) for model, value in values.items()}
    higher_is_better = entry.metric.higher_is_better

    # Ranked as the cells are printed rather than as the values behind them: two models whose
    # cells read the same are marked the same, and the reader can see why. A metric with no
    # better direction, and a row holding a single model, rank nothing and mark nothing.
    ranked: list[str] = []
    if higher_is_better is not None and len(formatted) > 1:
        ranked = sorted(set(formatted.values()), key=float, reverse=higher_is_better)

    # `None` where there is no such place to take: every model tied at the best leaves no second.
    best = ranked[0] if ranked else None
    second = ranked[1] if len(ranked) > 1 else None

    def marked(cell: str) -> str:
        if cell == best:
            return f'\\textbf{{{cell}}}'
        if cell == second:
            return f'\\underline{{{cell}}}'
        return cell

    return [MISSING if model not in formatted else marked(formatted[model]) for model in models]


def _escape_latex(text: str) -> str:
    """Escape the characters a dataset name or a model's label could carry into LaTeX text mode."""
    for character in ('\\', '&', '%', '$', '#', '_', '{', '}'):
        text = text.replace(character, f'\\{character}')
    return text


def latex_table(frame: pd.DataFrame, table: Table) -> str:
    """Render one table as booktabs, ready to be input into a paper.

    Args:
        frame: Every report read, from `read_reports`. Only the overall rows are tabulated: a
            table compares runs over their whole split.
        table: Which table to render.
    Returns:
        The tabular alone, with one column per model, the best value of each row in bold, the
        second best underlined and a rule between metrics. Needs the `booktabs`, `multirow` and
        `tabularx` packages, and belongs inside the paper's own float, which is where its caption
        and label are written.
    """
    overall = frame[frame['axis'] == table.axis]
    # One column per model and one row per log, in the order they are declared, so two tables of
    # the same runs read the same way.
    models = labels.MODELS.ordered(overall['model'])
    datasets = labels.DATASETS.ordered(overall['dataset'])
    header_row = '  Metric & Dataset & ' + ' & '.join(
        _escape_latex(labels.MODELS[model].label) for model in models
    )

    lines = ['\\toprule', header_row + ' \\\\', '\\midrule']
    for index, entry in enumerate(table.metrics):
        if index > 0:
            lines.append('\\midrule')
        # `=` rather than `*`: the Metric column is a fixed-width `X` column, and `multirow` only
        # wraps its label to that width, instead of overflowing past it, when told to match it.
        lines.append(f'  \\multirow{{{len(datasets)}}}{{=}}{{{entry.table_header}}}')
        scored = overall[overall['metric'] == entry.key]
        for dataset in datasets:
            values = scored[scored['dataset'] == dataset].set_index('model')['value']
            cells = _cells(entry, values.to_dict(), models)
            label = _escape_latex(labels.DATASETS[dataset])
            lines.append('   & ' + ' & '.join([label, *cells]) + ' \\\\')
    lines.append('\\bottomrule')

    text_columns = '*{2}{>{\\raggedright\\arraybackslash}X}'
    value_columns = f'*{{{len(models)}}}{{>{{\\centering\\arraybackslash}}X}}'
    preamble = f'\\begin{{tabularx}}{{\\linewidth}}{{{text_columns}|{value_columns}}}'
    return '\n'.join((preamble, *lines, '\\end{tabularx}')) + '\n'
