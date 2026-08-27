from collections.abc import Container, Mapping, Sequence

import pandas as pd

from src.visualization import labels
from src.visualization.catalogue import MetricEntry, Table

# What a cell reads where a model was never scored on a dataset.
MISSING = '-'


def _cells(
    entry: MetricEntry,
    values: Mapping[str, float],
    models: Sequence[str],
    best: Container[str],
) -> list[str]:
    """One row of a block: a formatted cell per model, the best group in bold.

    Args:
        entry: Which metric the row reports.
        values: What each model scored on it, for the models that were scored at all.
        models: The columns to fill, in the order the table writes them.
        best: The models whose score is the best or is indistinguishable from it, from
            `test_significance`. Empty where nothing is to be marked, e.g. a log holding a single
            model.
    Returns:
        One cell per model, `MISSING` where it was never scored.
    """
    formatted = {model: entry.format(value) for model, value in values.items()}
    return [
        MISSING
        if model not in formatted
        else f'\\textbf{{{formatted[model]}}}'
        if model in best
        else formatted[model]
        for model in models
    ]


def _escape_latex(text: str) -> str:
    """Escape the characters a dataset name or a model's label could carry into LaTeX text mode."""
    for character in ('\\', '&', '%', '$', '#', '_', '{', '}'):
        text = text.replace(character, f'\\{character}')
    return text


def latex_table(frame: pd.DataFrame, table: Table, significance: pd.DataFrame) -> str:
    """Render one table as booktabs, ready to be input into a paper.

    Args:
        frame: Every report read, from `read_reports`. Only the overall rows are tabulated: a
            table compares runs over their whole split.
        table: Which table to render.
        significance: Which models are the best or tied with it, from `test_significance`, under
            the same names as `frame`.
    Returns:
        The tabular alone, with one column per model, a rule between metrics, and the best value of
        each row in bold along with every value the test cannot separate from it. Needs the
        `booktabs`, `multirow` and `tabularx` packages, and belongs inside the paper's own float,
        which is where its caption and label are written.
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
        tied = significance[significance['metric'] == entry.key]
        for dataset in datasets:
            values = scored[scored['dataset'] == dataset].set_index('model')['value']
            marked = tied[(tied['dataset'] == dataset) & tied['best']]['model']
            cells = _cells(entry, values.to_dict(), models, set(marked))
            label = _escape_latex(labels.DATASETS[dataset])
            lines.append('   & ' + ' & '.join([label, *cells]) + ' \\\\')
    lines.append('\\bottomrule')

    text_columns = '*{2}{>{\\raggedright\\arraybackslash}X}'
    value_columns = f'*{{{len(models)}}}{{>{{\\centering\\arraybackslash}}X}}'
    preamble = f'\\begin{{tabularx}}{{\\linewidth}}{{{text_columns}|{value_columns}}}'
    return '\n'.join((preamble, *lines, '\\end{tabularx}')) + '\n'
