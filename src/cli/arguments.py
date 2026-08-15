import argparse
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
