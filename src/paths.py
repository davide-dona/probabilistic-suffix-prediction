from enum import StrEnum
from pathlib import Path

from src.identity import RunIdentity

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / 'config'
DATA_DIR = ROOT / 'data'
OUTPUTS_DIR = ROOT / 'outputs'
BEST_MODELS_DIR = ROOT / 'best-models'

# The layer every config is merged over, dataset- and hardware-agnostic.
BASE_CONFIG = CONFIG_DIR / 'base.yaml'


class Split(StrEnum):
    """The three splits preprocessing cuts a log into, named as their files on disk are.

    Iterating the members is what says a dataset has been preprocessed and what a pipeline
    reads, so the three names are written down here and nowhere else.
    """

    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


def hardware_config_path(hardware: str) -> Path:
    """One hardware profile, the layer a config is merged over between base and the dataset.

    Args:
        hardware: The profile's name, as passed to `-w`/`--hardware`, e.g. `mps`.
    Returns:
        The path to that profile's file.
    """
    return CONFIG_DIR / 'hardware' / f'{hardware}.yaml'


def dataset_config_path(config: str) -> Path:
    """One dataset's experiment config, the layer a config is merged over on top of the base
    config (and, for hardware-dependent pipelines, the hardware profile).

    Args:
        config: The config's name, as passed to `-c`/`--config`, e.g. `bpic17`.
    Returns:
        The path to that config's file.
    """
    return CONFIG_DIR / 'datasets' / f'{config}.yaml'


def _run_dir(run: RunIdentity) -> Path:
    """Where a run's files sit, relative to whichever `outputs/` directory holds them.

    The model sits above the tag, so one log's directory lists the models compared on it before
    any filename is read, and the leaf is the tag alone, since the directory it sits in already
    says which model wrote it.
    """
    return Path(run.dataset) / run.model / run.tag


def original_log(dataset: str) -> Path:
    """The raw log a dataset starts from, the one file preprocessing reads."""
    return DATA_DIR / dataset / 'original.csv'


def split_path(dataset: str, split: Split) -> Path:
    """One preprocessed split of a dataset.

    Args:
        dataset: The dataset the split belongs to.
        split: Which of the three to name.
    Returns:
        The path to that split's file.
    """
    return DATA_DIR / dataset / 'processed' / f'{split}.csv'


def codec_path(dataset: str) -> Path:
    """A dataset's fitted codec, separate from the splits it was fit on."""
    return DATA_DIR / dataset / 'codec' / 'dataset.json'


def declare_model_path(dataset: str) -> Path:
    """A dataset's declarative model, discovered from its train split at preprocessing time."""
    return DATA_DIR / dataset / 'declare' / 'model.decl'


def require_dataset(dataset: str) -> None:
    """Check that everything training and generation read from preprocessing is on disk, and say
    what is missing if it is not.

    Args:
        dataset: The dataset to check.
    Raises:
        FileNotFoundError: If any preprocessing output is missing, naming every one of them.
    """
    outputs = [split_path(dataset=dataset, split=split) for split in Split]
    outputs.append(codec_path(dataset))

    missing = [output for output in outputs if not output.exists()]
    if missing:
        raise FileNotFoundError(
            f'"{dataset}" has not been preprocessed: '
            f'{", ".join(str(output) for output in missing)} '
            f'{"are" if len(missing) > 1 else "is"} missing. Run '
            f'"uv run python -m pipelines.preprocess -c <config>" first.'
        )


def require_declare_model(dataset: str) -> None:
    """Check that a dataset's declarative model is on disk, since discovery is optional at
    preprocessing time and evaluation is the only reader.

    Args:
        dataset: The dataset to check.
    Raises:
        FileNotFoundError: If the declarative model is missing.
    """
    path = declare_model_path(dataset)
    if not path.exists():
        raise FileNotFoundError(
            f'"{dataset}" has no declarative model at {path}. Run '
            f'"uv run python -m pipelines.preprocess -c <config>" without --skip-discovery first.'
        )


def tensorboard_dir(run: RunIdentity) -> Path:
    """A run's TensorBoard events. One directory is one run, so the tag has to tell two runs of
    one model on one dataset apart."""
    return OUTPUTS_DIR / 'tensorboard' / _run_dir(run)


def checkpoint_path(run: RunIdentity) -> Path:
    """A run's last validated step, overwritten every validation; what `--resume` reads."""
    return OUTPUTS_DIR / 'checkpoints' / f'{_run_dir(run)}.pt'


def best_model_path(run: RunIdentity) -> Path:
    """A run's best step, overwritten whenever the selection score improves.

    Kept outside `outputs/` since these are the checkpoints published to the Hugging Face repo
    (`pipelines/publish.py`), not disposable run output.
    """
    return BEST_MODELS_DIR / f'{_run_dir(run)}.pt'


def generations_path(run: RunIdentity) -> Path:
    """The suffixes a run generated for the test split."""
    return OUTPUTS_DIR / 'generations' / f'{_run_dir(run)}.parquet'


def evaluation_path(run: RunIdentity) -> Path:
    """The report those generations scored."""
    return OUTPUTS_DIR / 'eval' / f'{_run_dir(run)}.json'


def plot_path(dataset: str, name: str, image_format: str) -> Path:
    """One figure, under the log whose prefixes it breaks down.

    Args:
        dataset: The log the figure describes.
        name: What the figure shows, e.g. a metric's name.
        image_format: The extension to write, e.g. `pdf`.
    Returns:
        The path to that figure in that format.
    """
    return OUTPUTS_DIR / 'plots' / dataset / f'{name}.{image_format}'


def comparison_table_path(name: str) -> Path:
    """One results table comparing every visualized run.

    Args:
        name: Which table, e.g. `comparison-point`.
    Returns:
        The path to that table's LaTeX source.
    """
    return OUTPUTS_DIR / 'plots' / f'{name}.tex'
