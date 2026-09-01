import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from src import paths
from src.cli import banner, step
from src.evaluation.bootstrap import INTERVAL_COLUMNS, read_intervals
from src.evaluation.report import Axis, read_reports
from src.visualization import (
    FIGURES,
    TABLES,
    apply_style,
    compose_figure,
    distribution_grid,
    embed_suffixes,
    latex_table,
    reported_models,
    test_significance,
)

# Which metrics the figures draw, so the per-prefix scores are read for those alone. Named off the
# catalogue rather than listed, so a metric added to a figure reaches the read with no change here.
_BANDED_METRICS = tuple(dict.fromkeys(entry.key for plot in FIGURES for entry in plot.metrics))
# Which breakdowns are bounded: the two a figure draws a line along. The overall one is not one of
# them, `read_intervals` saying why.
_BANDED_AXES = (Axis.PREFIX, Axis.SUFFIX)
# What a band joins onto its mean on, the key columns the two frames share.
_INTERVAL_KEYS = [key for key in INTERVAL_COLUMNS if key not in ('low', 'high')]


def _save_figure(figure: Figure, path: Path) -> None:
    """Write one finished figure and close it.

    Args:
        figure: The figure to write, closed afterwards so a run drawing dozens does not hold them
            all open.
        path: Where to write it, from `paths.FIGURE.prepare`.
    """
    figure.savefig(path)
    plt.close(figure)


def _draw_figures(frame: pd.DataFrame) -> int:
    """Draw every figure of the catalogue, each covering every log the reports cover at once.

    Args:
        frame: Every report read, from `read_reports`, with the bounds of `read_intervals` joined
            onto it, so a figure reads one frame and draws a line and its band off the same rows.
    Returns:
        How many figures were written, under `outputs/visual/figures/`.
    """
    written = 0
    for plot in FIGURES:
        _save_figure(
            figure=compose_figure(frame[frame['axis'].isin(plot.breakdowns)], plot),
            path=paths.FIGURE.prepare(plot.name),
        )
        written += 1
    return written


def _write_tables(frame: pd.DataFrame, significance: pd.DataFrame) -> int:
    """Write every comparison table, over every log at once, under `outputs/visual/tables/`.

    Args:
        frame: Every report read, from `read_reports`.
        significance: Which models are tied with the best of each row, from `test_significance`.
    Returns:
        How many tables were written.
    """
    for table in TABLES:
        paths.TABLE.prepare(table.name).write_text(latex_table(frame, table, significance))
    return len(TABLES)


def run(evaluation_files: Sequence[Path], generation_files: Sequence[Path]) -> None:
    """Draw a set of evaluation reports and tabulate them, under `outputs/visual/`.

    Args:
        evaluation_files: The reports to compare, from `python -m pipelines.evaluate`. These draw
            the metric figures and the comparison tables, and the per-prefix scores beside each of
            them are what the figures' bands are bounded by and what the tables' emphasis is tested
            on.
        generation_files: The generations of the same runs, from `python -m pipelines.generate`,
            or none. These draw the distribution figure, which costs minutes per log.
    Raises:
        ValueError: If a file is not what it should be, if a report has no per-prefix scores beside
            it, if a model has no look declared in `src.visualization.labels`, or if one log is
            given two runs of the same model.
    """
    apply_style()
    banner(
        'Drawing the figures and tables',
        {
            'reports': f'{len(evaluation_files)} file(s), with their per-prefix scores beside them',
            'generations': f'{len(generation_files)} file(s)' if generation_files else 'none',
            'figures': paths.FIGURES_DIR,
            'tables': paths.TABLES_DIR,
        },
    )

    with step(f'Reading {len(evaluation_files)} evaluation report(s)'):
        # Models sharing a style are one model from here on: one line, one column, one legend key.
        reports = reported_models(read_reports(evaluation_files))

    # Over the per-prefix scores beside each report, like the test below: a report holds the mean of
    # each metric at each length and says nothing about how well that many prefixes pin it down.
    # Joined on before anything is drawn, left, so a row with no bounds keeps its line and loses
    # only its band, and a figure still reads one frame.
    with step('Bounding how far each length’s mean could be off'):
        intervals = reported_models(
            read_intervals(evaluation_files, metrics=_BANDED_METRICS, axes=_BANDED_AXES)
        )
        banded = reports.merge(intervals, how='left', on=_INTERVAL_KEYS)

    logs = sorted(set(reports['dataset']))
    with step(f'Drawing {", ".join(logs)}'):
        drawn = _draw_figures(banded)

    # Over the per-prefix scores beside each report, since a mean cannot say whether two models
    # differ. A paired bootstrap over the cases of each log, which is seconds per log.
    with step('Testing which differences are real'):
        significance = reported_models(test_significance(evaluation_files))

    with step('Writing the comparison tables'):
        tables = _write_tables(reports, significance)

    # Opt-in, since this reads the generations rather than the reports and costs a minute or two
    # per log. Drawn from its own frame rather than through the catalogue above, since it reads
    # embeddings rather than report rows.
    if generation_files:
        with step(f'Embedding the suffixes of {len(generation_files)} run(s)'):
            embedding = reported_models(embed_suffixes(generation_files))
        with step('Drawing the generated distributions'):
            _save_figure(
                figure=distribution_grid(embedding),
                path=paths.FIGURE.prepare('distribution'),
            )
            drawn += 1

    print(
        f'\nWrote {drawn} figures in pdf to {paths.FIGURES_DIR} '
        f'and {tables} tables in tex to {paths.TABLES_DIR}'
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Plot and tabulate a set of evaluation reports for a paper.'
    )
    # Either named file by file or swept out of a directory. Both at once could only name one
    # file twice.
    reports = parser.add_mutually_exclusive_group(required=True)
    reports.add_argument(
        '-e',
        '--evaluations',
        type=paths.existing_file,
        metavar='REPORT',
        nargs='+',
        help='Paths to the evaluation reports to compare, from `pipelines.evaluate`. These draw '
        'the metric figures and the comparison tables, and the per-prefix scores written beside '
        "each of them are what the tables' emphasis is tested on.",
    )
    reports.add_argument(
        '-E',
        '--evaluations-dir',
        type=paths.existing_directory,
        metavar='DIR',
        nargs='+',
        help='Path(s) to a directory to compare every evaluation report under, at any depth, '
        'e.g. `outputs/eval` for all of them or `outputs/eval/bpic17` for one log. Several '
        'directories are swept together, e.g. `outputs/eval pinned/eval` to compare '
        'in-progress runs against pinned ones. Each report says which model and log it belongs '
        'to.',
    )

    generations = parser.add_mutually_exclusive_group()
    generations.add_argument(
        '-g',
        '--generations',
        type=paths.existing_file,
        metavar='GENERATIONS',
        nargs='+',
        help='Paths to the generations of the same runs, from `pipelines.generate`. These draw '
        'the UMAP figure of what each model generates against the ground truth, which costs a '
        'minute or two per log. Left out, every other figure is still drawn.',
    )
    generations.add_argument(
        '-G',
        '--generations-dir',
        type=paths.existing_directory,
        metavar='DIR',
        nargs='+',
        help='Path(s) to a directory to draw every generations file under, at any depth, e.g. '
        '`outputs/generations` for all of them or `outputs/generations/bpic17` for one log. '
        'Several directories are swept together, e.g. `outputs/generations pinned/generations`.',
    )
    args = parser.parse_args()

    evaluations = args.evaluations or []
    if args.evaluations_dir is not None:
        evaluations = paths.EVALUATION.sweep(args.evaluations_dir)

    generated = args.generations or []
    if args.generations_dir is not None:
        generated = paths.GENERATIONS.sweep(args.generations_dir)

    run(evaluations, generated)


if __name__ == '__main__':
    main()
