from collections.abc import Iterable

# What a log is called, as against `data.name` in its config, which is what its run directory is
# called. A log with nothing declared for it stops the run. Also the order logs are tabulated in,
# which `ordered_datasets` applies.
DATASETS: dict[str, str] = {
    'bpic17': 'BPIC17',
    'bpic19': 'BPIC19',
    'sepsis': 'Sepsis',
}


def dataset_label(dataset: str) -> str:
    """What a figure or a table calls one log.

    Args:
        dataset: Its name, as `data.name` in its config.
    Returns:
        Its declared label.
    Raises:
        ValueError: If nothing is declared for it, since a log printed under its own directory
            name would be the only one in the paper not written the way the log is published.
    """
    if dataset not in DATASETS:
        raise ValueError(
            f'nothing declared for the dataset {dataset!r}. Add it to DATASETS in '
            f'src/visualization/labels/datasets.py. The datasets declared are '
            f'{", ".join(DATASETS)}.'
        )
    return DATASETS[dataset]


def ordered_datasets(datasets: Iterable[str]) -> list[str]:
    """Sort a set of logs into the order they are drawn in, which `DATASETS` declares.

    Args:
        datasets: The logs present, in any order.
    Returns:
        Them alone, in the declared order, so a table's rows and a figure's keys read the same way
        however the reports were named on the command line.
    Raises:
        ValueError: If one of them is not declared.
    """
    present = set(datasets)
    for dataset in present:
        dataset_label(dataset)
    return [dataset for dataset in DATASETS if dataset in present]
