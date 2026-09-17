"""Self-contained inference checkpoints, atomically replaced on improvement."""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.identity import RunIdentity

MODEL_KEYS = ('config', 'model_state_dict')
CHECKPOINT_KEYS = (
    *MODEL_KEYS,
    'run',
    'step',
    'selection_score',
    'selection_metric',
    'selection_direction',
)

NUMPY_SAFE_GLOBALS = (
    np._core.multiarray.scalar,
    np.dtype,
    *(type(np.dtype(value)) for value in np.sctypeDict.values()),
)


def require_keys(
    checkpoint: dict, keys: Iterable[str], *, subject: str = 'checkpoint', purpose: str, remedy: str
) -> None:
    missing = [key for key in keys if key not in checkpoint]
    if missing:
        raise ValueError(
            f'{subject} is missing {", ".join(missing)}; cannot be {purpose}. {remedy}'
        )


def save_checkpoint(
    model: nn.Module,
    *,
    config: dict,
    step: int,
    selection_score: float,
    wandb_id: str | None,
    run: RunIdentity,
    path: Path,
) -> Path:
    temp = path.with_suffix('.pt.tmp')
    torch.save(
        obj={
            'config': config,
            'run': run.as_dict(),
            'model_state_dict': model.state_dict(),
            'step': step,
            'selection_score': selection_score,
            'selection_metric': 'energy_score_dls',
            'selection_direction': 'min',
            'wandb_id': wandb_id,
        },
        f=temp,
    )
    temp.replace(path)
    return path


def load_checkpoint(model_path: str | Path) -> dict:
    model_path = Path(model_path)
    with torch.serialization.safe_globals(NUMPY_SAFE_GLOBALS):
        checkpoint = torch.load(f=model_path, map_location='cpu', weights_only=True)
    if 'run' not in checkpoint:
        raise ValueError(
            f'{model_path} predates stable run identity. Train a new checkpoint with the current '
            'pipeline.'
        )
    require_keys(checkpoint, CHECKPOINT_KEYS, purpose='loaded', remedy='Train a new checkpoint.')
    model = checkpoint.get('config', {}).get('model', {})
    data = checkpoint.get('config', {}).get('data', {})
    run = RunIdentity.from_dict(checkpoint['run'])
    if run.dataset != data.get('name') or run.model != model.get('name'):
        raise ValueError('Checkpoint run identity does not match its training configuration')
    if checkpoint['selection_metric'] != 'energy_score_dls' or model.get('kind') not in (
        'transformer_cvae',
        'head_sampling_transformer',
    ):
        raise ValueError(
            'Checkpoint uses the legacy metric or model schema. Train a new checkpoint with '
            'transformer_cvae or head_sampling_transformer.'
        )
    return checkpoint


def checkpoint_identity(checkpoint: dict) -> RunIdentity:
    return RunIdentity.from_dict(checkpoint['run'])
