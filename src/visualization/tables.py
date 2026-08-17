from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from src.evaluation.report import Axis
from src.visualization import labels

# What a cell reads where a model was never scored on a dataset.
MISSING = '-'


@dataclass(frozen=True)
class Table:
    """One comparison table: what its file is called and which metrics it puts in rows.

    A row carries its own header, since a table holding a single estimator throughout makes the
    qualifier a metric's label needs elsewhere redundant.
    """

    name: str
    rows: tuple[tuple[str, str], ...]  # `(metric, header)`, one block each, in the order written


# The two tables answer two questions, and each of them holds a single estimator so that the
# emphasis compares models rather than a model against itself. A point estimate beats the mean of
# ten stochastic draws almost by construction, so the two must never share a row.
POINT_TABLE = Table(
    name='comparison-point',
    rows=(
        ('dls_point', 'Suffix DLS'),
        ('conformance_point', 'Suffix conformance'),
        ('length_ae_point', 'Suffix length MAE'),
        ('remaining_time_ae_point_days', 'Remaining time MAE'),
    ),
)

PROBABILISTIC_TABLE = Table(
    name='comparison-probabilistic',
    rows=(
        ('energy_distance', 'Energy distance'),
        ('hit_rate_at_1', 'Hit rate @1'),
        ('hit_rate_at_5', 'Hit rate @5'),
        ('hit_rate_at_10', 'Hit rate @10'),
        ('dls_best', 'Best-of-k DLS'),
        ('dls_mean', 'Mean DLS'),
        ('conformance_mean', 'Mean conformance'),
    ),
)

TABLES = (POINT_TABLE, PROBABILISTIC_TABLE)


def _cells(metric: str, values: Mapping[str, float], models: Sequence[str]) -> list[str]:
    """One row of a block: a formatted cell per model, the best of them in bold."""
    spec = labels.metric(metric)
    formatted = {model: spec.format(value) for model, value in values.items()}

    best: set[str] = set()
    if spec.higher_is_better is not None and len(formatted) > 1:
        winner = (max if spec.higher_is_better else min)(formatted.values(), key=float)
        best = {model for model, cell in formatted.items() if cell == winner}

    return [
        MISSING
        if model not in formatted
        else (f'\\textbf{{{formatted[model]}}}' if model in best else formatted[model])
        for model in models
    ]


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
        The tabular alone, with one column per model, the best value of each row in bold and a
        rule between metrics. Needs the `booktabs`, `multirow` and `tabularx` packages, and
        belongs inside the paper's own float, which is where its caption and label are written.
    """
    overall = frame[frame['axis'] == Axis.OVERALL]
    # One column per model and one row per log, in the order they are declared, so two tables of
    # the same runs read the same way.
    models = labels.ordered_models(overall['model'])
    datasets = labels.ordered_datasets(overall['dataset'])
    header_row = '  Metric & Dataset & ' + ' & '.join(
        _escape_latex(labels.model_style(model).label) for model in models
    )

    lines = ['\\toprule', header_row + ' \\\\', '\\midrule']
    for index, (metric, header) in enumerate(table.rows):
        if index > 0:
            lines.append('\\midrule')
        # `=` rather than `*`: the Metric column is a fixed-width `X` column, and `multirow` only
        # wraps its label to that width, instead of overflowing past it, when told to match it.
        cell = labels.metric(metric).table_header(header)
        lines.append(f'  \\multirow{{{len(datasets)}}}{{=}}{{{cell}}}')
        scored = overall[overall['metric'] == metric]
        for dataset in datasets:
            values = scored[scored['dataset'] == dataset].set_index('model')['value']
            cells = _cells(metric, values.to_dict(), models)
            label = _escape_latex(labels.dataset_label(dataset))
            lines.append('   & ' + ' & '.join([label, *cells]) + ' \\\\')
    lines.append('\\bottomrule')

    text_columns = '*{2}{>{\\raggedright\\arraybackslash}X}'
    value_columns = f'*{{{len(models)}}}{{>{{\\centering\\arraybackslash}}X}}'
    preamble = f'\\begin{{tabularx}}{{\\linewidth}}{{{text_columns}|{value_columns}}}'
    return '\n'.join((preamble, *lines, '\\end{tabularx}')) + '\n'
