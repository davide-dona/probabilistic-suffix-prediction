from enum import StrEnum
from pathlib import Path

from src.identity import DatasetIdentity, RunIdentity

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
OUTPUTS_DIR = ROOT / 'outputs'


class Split(StrEnum):
    """The three splits preprocessing cuts a log into, named as their files on disk are.

    Iterating the members is what says a dataset has been preprocessed and what a pipeline
    reads, so the three names are written down here and nowhere else.
    """

    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


def _dataset_dir(dataset: DatasetIdentity) -> Path:
    """Where a dataset's own files sit, relative to whichever directory holds them."""
    return Path(dataset.name) / dataset.variant


def _run_dir(run: RunIdentity) -> Path:
    """Where a run's files sit, relative to whichever `outputs/` directory holds them.

    The model sits above the variant, so one log's directory lists the models compared on it
    before any filename is read, and the leaf is the tag alone, since the directory it sits in
    already says which model wrote it.
    """
    return Path(run.dataset.name) / run.model / run.dataset.variant / run.tag


def original_log(dataset: DatasetIdentity) -> Path:
    """The raw log a dataset starts from, the one file preprocessing reads.

    Named by log rather than by variant: every description of a log reads the same events.
    """
    return DATA_DIR / dataset.name / 'original.csv'


def split_path(dataset: DatasetIdentity, split: Split) -> Path:
    """One preprocessed split of a dataset.

    Args:
        dataset: The dataset the split belongs to.
        split: Which of the three to name.
    Returns:
        The path to that split's file.
    """
    return DATA_DIR / _dataset_dir(dataset) / 'processed' / f'{split}.csv'


def codec_path(dataset: DatasetIdentity) -> Path:
    """A dataset's fitted codec, next to the splits it was fit on."""
    return DATA_DIR / _dataset_dir(dataset) / 'processed' / 'dataset.json'


def declare_model_path(dataset: DatasetIdentity) -> Path:
    """A dataset's declarative model, discovered from its train split at preprocessing time."""
    return DATA_DIR / _dataset_dir(dataset) / 'declare' / 'model.decl'


def require_dataset(dataset: DatasetIdentity) -> None:
    """Check that everything preprocessing produces is on disk, and say what is missing if it
    is not.

    Args:
        dataset: The dataset to check.
    Raises:
        FileNotFoundError: If any preprocessing output is missing, naming every one of them.
    """
    outputs = [split_path(dataset=dataset, split=split) for split in Split]
    outputs.append(codec_path(dataset))
    outputs.append(declare_model_path(dataset))

    missing = [output for output in outputs if not output.exists()]
    if missing:
        raise FileNotFoundError(
            f'"{dataset}" has not been preprocessed: '
            f'{", ".join(str(output) for output in missing)} '
            f'{"are" if len(missing) > 1 else "is"} missing. Run '
            f'"uv run python -m pipelines.preprocess -c <config>" first.'
        )


def tensorboard_dir(run: RunIdentity) -> Path:
    """A run's TensorBoard events. One directory is one run, so the tag has to tell two runs of
    one model on one variant apart."""
    return OUTPUTS_DIR / 'tensorboard' / _run_dir(run)


def checkpoint_path(run: RunIdentity) -> Path:
    """A run's last validated step, overwritten every validation; what `--resume` reads."""
    return OUTPUTS_DIR / 'checkpoints' / f'{_run_dir(run)}.pt'


def best_model_path(run: RunIdentity) -> Path:
    """A run's best step, overwritten whenever the selection score improves."""
    return OUTPUTS_DIR / 'best-models' / f'{_run_dir(run)}.pt'


def generations_path(run: RunIdentity) -> Path:
    """The suffixes a run generated for the test split."""
    return OUTPUTS_DIR / 'generations' / f'{_run_dir(run)}.parquet'


def evaluation_path(run: RunIdentity) -> Path:
    """The report those generations scored."""
    return OUTPUTS_DIR / 'eval' / f'{_run_dir(run)}.json'


def plot_path(dataset: str, name: str, image_format: str) -> Path:
    """One figure, under the log whose prefixes it breaks down.

    Args:
        dataset: The log the figure describes, without a variant: one figure holds every variant
            of a log, since they are the same process seen through different preprocessing.
        name: What the figure shows, e.g. a metric's name.
        image_format: The extension to write, e.g. `pdf`.
    Returns:
        The path to that figure in that format.
    """
    return OUTPUTS_DIR / 'plots' / dataset / f'{name}.{image_format}'


def comparison_table_path(name: str, table_format: str) -> Path:
    """One results table comparing every visualized run.

    Args:
        name: Which table, e.g. `comparison-quality`.
        table_format: The extension to write, e.g. `tex`.
    Returns:
        The path to that table in that format.
    """
    return OUTPUTS_DIR / 'plots' / f'{name}.{table_format}'
