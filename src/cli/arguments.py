import argparse
from collections.abc import Sequence
from pathlib import Path


def existing_file(value: str) -> Path:
    """Read a path argument that has to already exist.

    Args:
        value: What the flag was given.
    Returns:
        That path.
    Raises:
        argparse.ArgumentTypeError: If nothing is there, so a mistyped path fails while the
            arguments are being read rather than part-way into whatever reads the file.
    """
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f'no such file: {path}')
    return path


def existing_directory(value: str) -> Path:
    """Read a path argument that has to already be a directory.

    Args:
        value: What the flag was given.
    Returns:
        That path.
    Raises:
        argparse.ArgumentTypeError: If nothing is there or a file is, so a mistyped path fails
            while the arguments are being read rather than part-way into whatever walks it.
    """
    path = Path(value)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f'no such directory: {path}')
    return path


def swept(directories: Sequence[Path], suffix: str, kind: str, pipeline: str) -> list[Path]:
    """Find every artifact of one kind under one or more directories, e.g. `outputs/eval` and
    `pinned/eval` swept together to compare in-progress runs against pinned ones.

    Args:
        directories: The directories to sweep.
        suffix: The extension the artifact is written with, e.g. `.json`.
        kind: What the artifact is called in the error, e.g. `evaluation reports`.
        pipeline: The pipeline that writes it, named in the error.
    Returns:
        The artifacts under them, in path order.
    Raises:
        FileNotFoundError: If they hold none at all.
    """
    found = sorted(path for directory in directories for path in directory.rglob(f'*{suffix}'))
    if not found:
        under = ', '.join(str(directory) for directory in directories)
        raise FileNotFoundError(
            f'no {kind} under {under}. Run `python -m {pipeline}` first, or sweep the right '
            'directory.'
        )
    return found


def add_config_argument(container: argparse._ActionsContainer, *, required: bool) -> None:
    """Add `-c`/`--config`, the path to a dataset's experiment config.

    Args:
        container: What to add the argument to. A group rather than a parser where the config is
            one of several mutually exclusive ways to describe a run.
        required: Whether the config has to be named. False inside a mutually exclusive group,
            which argparse requires as a whole rather than one member at a time.
    """
    container.add_argument(
        '-c',
        '--config',
        type=existing_file,
        metavar='CONFIG',
        required=required,
        help="Path to this experiment's dataset config, e.g. config/datasets/bpic17.yaml.",
    )


def add_hardware_argument(parser: argparse.ArgumentParser, *, required: bool) -> None:
    """Add `-w`/`--hardware`, the path to the profile a run is sized for.

    Args:
        parser: The parser to add the argument to.
        required: Whether the profile has to be named. False where a run may instead be described
            by a checkpoint that already carries one.
    """
    parser.add_argument(
        '-w',
        '--hardware',
        type=existing_file,
        metavar='HARDWARE',
        required=required,
        help='Path to the hardware profile to run under, e.g. config/hardware/mps.yaml.',
    )
