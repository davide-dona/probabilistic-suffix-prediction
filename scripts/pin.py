import argparse
import shutil
from pathlib import Path

from src import paths
from src.cli import banner, existing_file
from src.identity import RunIdentity
from src.model import load_checkpoint


def _artifacts(run: RunIdentity) -> list[tuple[Path, Path]]:
    """Every path one run may have written under `outputs/`, paired with where it moves to
    under `pinned/`.

    Args:
        run: The run to pin.
    Returns:
        Source/destination pairs. A pair whose source does not exist is simply not this run's,
        since a run's evaluation report or TensorBoard log may not have been written yet.
    """
    return [
        (paths.best_model_path(run), paths.best_model_path(run, pinned=True)),
        (paths.checkpoint_path(run), paths.checkpoint_path(run, pinned=True)),
        (paths.generations_path(run), paths.generations_path(run, pinned=True)),
        (paths.evaluation_path(run), paths.evaluation_path(run, pinned=True)),
        (paths.tensorboard_dir(run), paths.tensorboard_dir(run, pinned=True)),
    ]


def run(checkpoint_paths: list[Path], *, force: bool) -> None:
    """Move a run's checkpoints, generations, evaluation report and TensorBoard log from
    `outputs/` to `pinned/`, so they survive a wipe of the former.

    Args:
        checkpoint_paths: The best checkpoints of the runs to pin, from
            `outputs/checkpoints/best/`. A run's identity is read from the checkpoint itself.
        force: Whether to overwrite a run already pinned, instead of refusing to.
    Raises:
        FileNotFoundError: If a checkpoint is missing.
        SystemExit: If a run is already pinned and `force` is false.
    """
    missing = [path for path in checkpoint_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f'no checkpoint at {", ".join(str(path) for path in missing)}.')

    identities = [RunIdentity.from_dict(load_checkpoint(path)['run']) for path in checkpoint_paths]

    already_pinned = [
        identity for identity in identities if paths.best_model_path(identity, pinned=True).exists()
    ]
    if already_pinned and not force:
        raise SystemExit(
            f'already pinned: {", ".join(str(identity) for identity in already_pinned)}. Pass '
            '--force to overwrite.'
        )

    banner('Pinning runs', {'runs': f'{len(identities)}', 'destination': paths.PINNED_DIR})

    moved = 0
    for identity in identities:
        for source, destination in _artifacts(identity):
            if not source.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                shutil.rmtree(destination) if destination.is_dir() else destination.unlink()
            shutil.move(str(source), str(destination))
            moved += 1
        print(f'  pinned {identity}')

    print(f'\nMoved {moved} artifact(s) into {paths.PINNED_DIR}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move a run's checkpoints, generations, evaluation report and TensorBoard "
        'log from outputs/ to pinned/, so it survives a wipe of outputs/.'
    )
    parser.add_argument(
        '-m',
        '--checkpoint',
        type=existing_file,
        metavar='CHECKPOINT',
        nargs='+',
        required=True,
        help="Path(s) to the run(s)' best checkpoint, from outputs/checkpoints/best/.",
    )
    parser.add_argument(
        '-f',
        '--force',
        action='store_true',
        help='Overwrite a run already pinned instead of refusing to.',
    )
    args = parser.parse_args()

    run(args.checkpoint, force=args.force)


if __name__ == '__main__':
    main()
