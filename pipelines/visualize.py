import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from src import paths
from src.visualization import (
    ACCURACY_METRICS,
    BY_PREFIX_LENGTH,
    BY_SUFFIX_LENGTH,
    ERROR_METRICS,
    METRICS,
    TABLES,
    PlottedRun,
    coverage_cutoff,
    latex_table,
    length_curve,
    load_runs,
    markdown_table,
    metric_grid,
    support_curve,
    use_paper_style,
)

IMAGE_FORMATS = ('pdf', 'svg', 'png')


def save_figure(figure: Figure, paths: Iterable[Path]) -> None:
    """Write a figure to every path asked for and close it.

    Args:
        figure: The finished figure. Closed here, since nothing reads it back.
        paths: Where to write it, one per format; the suffix of each decides the format.
    """
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path)
    plt.close(figure)


def _write_dataset_figures(
    dataset: str,
    runs: Sequence[PlottedRun],
    *,
    formats: Sequence[str],
    coverage: float,
) -> int:
    """Draw every figure of one log and write it under `outputs/plots/<dataset>/`.

    Args:
        dataset: The log the runs were scored on, which names the directory.
        runs: The runs to compare, drawn on the same axes.
        formats: The image formats to write each figure in.
        coverage: The share of prefix pairs the x-axis of the metric figures must cover.
    Returns:
        How many figures were drawn, whatever the number of formats each was written in.
    """

    def write(figure: Figure, name: str) -> None:
        save_figure(
            figure,
            [paths.plot_path(dataset, name, image_format) for image_format in formats],
        )

    figures = 0
    for axis, prefix in (
        (BY_PREFIX_LENGTH, 'by-prefix-length/'),
        (BY_SUFFIX_LENGTH, 'by-suffix-length/'),
    ):
        cutoff = coverage_cutoff(runs, coverage, axis)

        for spec in METRICS:
            write(length_curve(runs, spec, cutoff, axis), f'{prefix}{spec.key}')

        write(metric_grid(runs, ACCURACY_METRICS, cutoff, axis), f'{prefix}accuracy-grid')
        write(metric_grid(runs, ERROR_METRICS, cutoff, axis), f'{prefix}error-grid')
        write(support_curve(runs, cutoff, coverage, axis), f'{prefix}support')
        figures += len(METRICS) + 3

    return figures


def _dataset_labels(pairs: Sequence[str] | None) -> dict[str, str]:
    """Read the `name=Display` pairs a table's Dataset column is rendered through.

    Args:
        pairs: What the flag was given, or `None` for no renaming at all.
    Returns:
        The display name of each log named, keyed by the log's own name.
    Raises:
        ValueError: If a pair is not `name=Display`.
    """
    labels = {}
    for pair in pairs or ():
        name, separator, display = pair.partition('=')
        if not separator or not name or not display:
            raise ValueError(f'expected a dataset label as `name=Display`, got {pair!r}.')
        labels[name] = display
    return labels


def run(
    evaluation_files: Sequence[Path],
    labels: Sequence[str] | None,
    dataset_labels: Sequence[str] | None,
    formats: Sequence[str],
    coverage: float,
) -> None:
    """Draw a set of evaluation reports and tabulate them, under `outputs/plots/`.

    Args:
        evaluation_files: The reports to compare, from `python -m pipelines.evaluate`.
        labels: What to call each of them in legends and tables, in the same order, or `None` for
            each run's model and variant.
        dataset_labels: `name=Display` pairs renaming a log in the tables, e.g. `bpic17=BPIC17`.
            Only the tables: a figure directory keeps the log's own name.
        formats: The image formats to write each figure in.
        coverage: The share of prefix pairs the x-axis of a metric figure must cover, which bounds
            it short of the sparse tail of long prefixes. 1.0 draws every length.
    Raises:
        ValueError: If `coverage` is not a share of the pairs, or a dataset label is malformed.
    """
    if not 0.0 < coverage <= 1.0:
        raise ValueError(f'coverage must be a share of the pairs in (0, 1], got {coverage}.')

    displayed = _dataset_labels(dataset_labels)
    grouped = load_runs(evaluation_files, labels)
    use_paper_style()

    figures = 0
    for dataset, runs in grouped.items():
        figures += _write_dataset_figures(dataset, runs, formats=formats, coverage=coverage)
        print(f'Drew {dataset} from {len(runs)} run(s).', flush=True)

    # The tables see the display names, the figures the log's own: one is read by a person, the
    # other is a directory.
    tabulated = {displayed.get(dataset, dataset): runs for dataset, runs in grouped.items()}
    for table in TABLES:
        markdown = markdown_table(tabulated, table)
        for table_format, text in (('md', markdown), ('tex', latex_table(tabulated, table))):
            path = paths.comparison_table_path(table.name, table_format)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        print(f'\n{table.name}\n{markdown}', end='', flush=True)

    print(
        f'\nWrote {figures} figures in {", ".join(formats)} and {len(TABLES)} tables in md, tex '
        f'to {paths.comparison_table_path(TABLES[0].name, "md").parent}'
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Plot and tabulate a set of evaluation reports for a paper.'
    )
    parser.add_argument(
        '-e',
        '--evaluations',
        type=Path,
        nargs='+',
        required=True,
        help='Paths to the evaluation reports to compare, from `pipelines.evaluate`.',
    )
    parser.add_argument(
        '-l',
        '--labels',
        nargs='+',
        default=None,
        help='What to call each report in legends and tables, in the same order. Two reports '
        'sharing a label are one model on two datasets. Defaults to `<model> (<variant>)`.',
    )
    parser.add_argument(
        '--dataset-labels',
        nargs='+',
        default=None,
        help='How to name a dataset in the tables, as `name=Display` pairs, e.g. `bpic17=BPIC17`. '
        "Figures keep the dataset's own name, since it is a directory.",
    )
    parser.add_argument(
        '-f',
        '--formats',
        nargs='+',
        choices=IMAGE_FORMATS,
        default=['pdf'],
        help='Which image formats to write each figure in.',
    )
    parser.add_argument(
        '--coverage',
        type=float,
        default=0.95,
        help='The share of prefix pairs the x-axis must cover, which bounds it short of the '
        'sparse tail of long prefixes. 1.0 draws every length.',
    )
    args = parser.parse_args()

    run(args.evaluations, args.labels, args.dataset_labels, args.formats, args.coverage)


if __name__ == '__main__':
    main()
