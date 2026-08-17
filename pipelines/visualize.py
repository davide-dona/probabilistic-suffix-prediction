import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from src import paths
from src.cli import banner, existing_directory, existing_file, step, swept
from src.evaluation.report import read_reports
from src.visualization import (
    FIGURES,
    TABLES,
    apply_style,
    compose_figure,
    distribution_grid,
    embed_suffixes,
    latex_table,
    reported_models,
)


def _save_figure(figure: Figure, path: Path) -> None:
    """Write one finished figure and close it, creating its parent directories.

    Args:
        figure: The figure to write, closed afterwards so a run drawing dozens does not hold them
            all open.
        path: Where to write it, from `paths.figure_path`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def _draw_figures(frame: pd.DataFrame) -> int:
    """Draw every figure of the catalogue, for every log the reports cover.

    Args:
        frame: Every report read, from `read_reports`.
    Returns:
        How many figures were written, under `outputs/visual/figures/<dataset>/`.
    """
    written = 0
    for dataset, rows in frame.groupby('dataset', sort=False):
        for plot in FIGURES:
            _save_figure(
                figure=compose_figure(rows[rows['axis'] == plot.axis], plot),
                path=paths.figure_path(str(dataset), plot.name),
            )
            written += 1
    return written


def _write_tables(frame: pd.DataFrame) -> int:
    """Write every comparison table, over every log at once, under `outputs/visual/tables/`.

    Args:
        frame: Every report read, from `read_reports`.
    Returns:
        How many tables were written.
    """
    for table in TABLES:
        path = paths.comparison_table_path(table.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(latex_table(frame, table))
    return len(TABLES)


def run(evaluation_files: Sequence[Path], generation_files: Sequence[Path]) -> None:
    """Draw a set of evaluation reports and tabulate them, under `outputs/visual/`.

    Args:
        evaluation_files: The reports to compare, from `python -m pipelines.evaluate`. These draw
            the metric figures and the comparison tables.
        generation_files: The generations of the same runs, from `python -m pipelines.generate`,
            or none. These draw the distribution figure, which costs minutes per log.
    Raises:
        ValueError: If a file is not what it should be, if a model has no look declared in
            `src.visualization.labels`, or if one log is given two runs of the same model.
    """
    apply_style()
    banner(
        'Drawing the figures and tables',
        {
            'reports': f'{len(evaluation_files)} file(s)',
            'generations': f'{len(generation_files)} file(s)' if generation_files else 'none',
            'figures': paths.FIGURES_DIR,
            'tables': paths.TABLES_DIR,
        },
    )

    with step(f'Reading {len(evaluation_files)} evaluation report(s)'):
        # Models sharing a style are one model from here on: one line, one column, one legend key.
        reports = reported_models(read_reports(evaluation_files))

    logs = sorted(set(reports['dataset']))
    with step(f'Drawing {", ".join(logs)}'):
        drawn = _draw_figures(reports)

    with step('Writing the comparison tables'):
        tables = _write_tables(reports)

    # Opt-in, since this reads the generations rather than the reports and costs a minute or two
    # per log. One figure per log, so it is drawn here rather than through the catalogue above.
    if generation_files:
        with step(f'Embedding the suffixes of {len(generation_files)} run(s)'):
            embedding = reported_models(embed_suffixes(generation_files))
        with step('Drawing the generated distributions'):
            for dataset, rows in embedding.groupby('dataset', sort=False):
                _save_figure(
                    figure=distribution_grid(rows),
                    path=paths.figure_path(str(dataset), 'distribution'),
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
        type=existing_file,
        metavar='REPORT',
        nargs='+',
        help='Paths to the evaluation reports to compare, from `pipelines.evaluate`. These draw '
        'the metric figures and the comparison tables.',
    )
    reports.add_argument(
        '-E',
        '--evaluations-dir',
        type=existing_directory,
        metavar='DIR',
        help='Path to a directory to compare every evaluation report under, at any depth, e.g. '
        '`outputs/eval` for all of them or `outputs/eval/bpic17` for one log. Each report says '
        'which model and log it belongs to.',
    )

    generations = parser.add_mutually_exclusive_group()
    generations.add_argument(
        '-g',
        '--generations',
        type=existing_file,
        metavar='GENERATIONS',
        nargs='+',
        help='Paths to the generations of the same runs, from `pipelines.generate`. These draw '
        'the UMAP figure of what each model generates against the ground truth, which costs a '
        'minute or two per log. Left out, every other figure is still drawn.',
    )
    generations.add_argument(
        '-G',
        '--generations-dir',
        type=existing_directory,
        metavar='DIR',
        help='Path to a directory to draw every generations file under, at any depth, e.g. '
        '`outputs/generations` for all of them or `outputs/generations/bpic17` for one log.',
    )
    args = parser.parse_args()

    evaluations = args.evaluations or []
    if args.evaluations_dir is not None:
        evaluations = swept(
            args.evaluations_dir, '.json', 'evaluation reports', 'pipelines.evaluate'
        )

    generated = args.generations or []
    if args.generations_dir is not None:
        generated = swept(args.generations_dir, '.parquet', 'generations', 'pipelines.generate')

    run(evaluations, generated)


if __name__ == '__main__':
    main()
