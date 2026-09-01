from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from src.identity import RunIdentity

# What rebuilding the model a checkpoint holds reads, and so what `TransformerCVAE.from_checkpoint`
# refuses to guess at.
MODEL_KEYS = ('model_config', 'model_state_dict')

# What carrying a run on reads: the weights, and the optimizer, early-stopping and random state
# that put the run back where it left off. A checkpoint trimmed by `inference_payload` carries the
# first half of these and none of the second.
RESUME_KEYS = (
    'model_state_dict',
    'experiment_config',
    'run',
    'step',
    'selection_score',
    'optimizer_state',
    'early_stopping_state',
    'rng_state',
)

# What a checkpoint is worth keeping for once it will only ever be generated from: the two keys
# `TransformerCVAE.from_checkpoint` reads, the one `pipelines/generate.py` names its output after,
# and three that say which step of which run this is and how well it scored.
INFERENCE_KEYS = (
    'model_config',
    'model_state_dict',
    'run',
    'experiment_config',
    'step',
    'selection_score',
)


def require_keys(
    checkpoint: dict, keys: Iterable[str], *, subject: str = 'checkpoint', purpose: str, remedy: str
) -> None:
    """Check that a checkpoint carries everything one use of it reads.

    Args:
        checkpoint: What `load_checkpoint` read.
        keys: The keys that use reads, named in the error in the order given.
        subject: What the error calls the file, e.g. the path it was read from.
        purpose: What it was about to be used for, e.g. `resumed from`.
        remedy: What to do instead, one sentence.
    Raises:
        ValueError: If any key is missing, naming every one of them.
    """
    missing = [key for key in keys if key not in checkpoint]
    if missing:
        raise ValueError(
            f'{subject} is missing {", ".join(missing)}, so it cannot be {purpose}. {remedy}'
        )


def save_checkpoint(
    model: nn.Module,
    *,
    experiment_config: dict,
    step: int,
    selection_score: float,
    run: RunIdentity,
    optimizer_state: dict,
    early_stopping_state: dict,
    rng_state: dict,
    path: str | Path,
) -> Path:
    """
    Save a checkpoint holding everything a run needs to be rebuilt or picked back up.

    The run's config travels with the weights, so the same model can be rebuilt later without
    being told a single hyperparameter, and the rest of the training state travels with them,
    so `pipelines/train.py -r` can carry on from here rather than from scratch.

    Args:
        model: The model whose weights to save.
        experiment_config: The run's whole `ExperimentConfig`, dumped to plain data, so that
            resuming needs nothing but this file. Its `model` section is written out beside it,
            since that is all `TransformerCVAE.from_checkpoint` reads.
        step: The optimizer step the weights are from. The filename does not say, so the file
            has to.
        selection_score: That step's generation score, the number the best is chosen on.
        run: Which run these weights belong to, so a resumed run keeps writing to the same
            TensorBoard directory and the same files, and so the generations they produce are
            named after the run rather than after the file the weights were read from.
        optimizer_state: The optimizer's `state_dict`.
        early_stopping_state: The `EarlyStopper`'s `state_dict`, which carries the best score
            seen as well as the patience count.
        rng_state: The random streams, as captured by `training/train.py`'s `rng_state`.
        path: Where to write, its directory already made, from `paths.LAST_CHECKPOINT.prepare`
            or `paths.BEST_CHECKPOINT.prepare`.
    Returns:
        The path written to.
    """
    path = Path(path)

    # Written aside and moved into place, so a run interrupted mid-save leaves the last good
    # checkpoint intact rather than a truncated file where one is expected.
    temp_path = path.with_name(f'{path.name}.tmp')
    torch.save(
        obj={
            'model_config': experiment_config['model'],
            'model_state_dict': model.state_dict(),
            'step': step,
            'selection_score': selection_score,
            'experiment_config': experiment_config,
            'run': asdict(run),
            'optimizer_state': optimizer_state,
            'early_stopping_state': early_stopping_state,
            'rng_state': rng_state,
        },
        f=temp_path,
    )
    temp_path.replace(target=path)
    return path


def load_checkpoint(model_path: str | Path) -> dict:
    """Read a checkpoint file written by `save_checkpoint`."""
    return torch.load(f=Path(model_path), map_location='cpu', weights_only=False)


def inference_payload(checkpoint: dict) -> dict:
    """Trim a checkpoint to what generating from it needs.

    Args:
        checkpoint: What `load_checkpoint` read.
    Returns:
        A new dict holding only the keys inference and provenance need.
    Raises:
        ValueError: If the checkpoint is missing any of them, naming every one, so a file written
            before this format fails here rather than half-way through someone else's generation.
    """
    require_keys(
        checkpoint,
        INFERENCE_KEYS,
        purpose='published',
        remedy='It was written by an older version of `save_checkpoint`; retrain, or publish a '
        'newer run.',
    )
    return {key: checkpoint[key] for key in INFERENCE_KEYS}
