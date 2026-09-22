import json
import platform
import subprocess
import sys
from functools import cache
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.identity import RunIdentity, validate_dataset, validate_run_id


def _available(path: Path) -> Path:
    if path.exists():
        raise FileExistsError(f'Output directory already exists: {path}')
    return path


def _dataset_output(stage: str, dataset: str, run_id: str) -> str:
    validate_dataset(dataset)
    validate_run_id(run_id)
    return _available(Path('outputs') / stage / dataset / run_id).as_posix()


def _dataset_subdir(stage: str, dataset: str, run_id: str) -> str:
    validate_dataset(dataset)
    validate_run_id(run_id)
    path = _available(Path('outputs') / stage / dataset / run_id)
    return path.relative_to(Path('outputs') / stage).as_posix()


def _model_output(stage: str, dataset: str, model: str, run_id: str) -> str:
    run = RunIdentity(dataset=dataset, model=model, run_id=run_id)
    return _available(Path('outputs') / stage / str(run)).as_posix()


def _model_subdir(stage: str, dataset: str, model: str, run_id: str) -> str:
    run = RunIdentity(dataset=dataset, model=model, run_id=run_id)
    path = _available(Path('outputs') / stage / str(run))
    return path.relative_to(Path('outputs') / stage).as_posix()


@cache
def _checkpoint_identity(path: str) -> RunIdentity:
    from src.model import load_checkpoint

    return RunIdentity.from_dict(load_checkpoint(Path(path))['run'])


@cache
def _generations_identity(path: str) -> RunIdentity:
    from src.inference.generation_store import Generations

    with Generations(Path(path)) as generations:
        return generations.run


def _checkpoint_output(stage: str, path: str) -> str:
    run = _checkpoint_identity(path)
    return _model_output(stage, run.dataset, run.model, run.run_id)


def _checkpoint_subdir(stage: str, path: str) -> str:
    run = _checkpoint_identity(path)
    return _model_subdir(stage, run.dataset, run.model, run.run_id)


def _generations_output(stage: str, path: str) -> str:
    run = _generations_identity(path)
    return _model_output(stage, run.dataset, run.model, run.run_id)


def _generations_subdir(stage: str, path: str) -> str:
    run = _generations_identity(path)
    return _model_subdir(stage, run.dataset, run.model, run.run_id)


OmegaConf.register_new_resolver('dataset_output', _dataset_output, replace=True, use_cache=True)
OmegaConf.register_new_resolver('dataset_subdir', _dataset_subdir, replace=True, use_cache=True)
OmegaConf.register_new_resolver('model_output', _model_output, replace=True, use_cache=True)
OmegaConf.register_new_resolver('model_subdir', _model_subdir, replace=True, use_cache=True)
OmegaConf.register_new_resolver(
    'checkpoint_output',
    _checkpoint_output,
    replace=True,
    use_cache=True,
)
OmegaConf.register_new_resolver(
    'checkpoint_subdir',
    _checkpoint_subdir,
    replace=True,
    use_cache=True,
)
OmegaConf.register_new_resolver(
    'generations_output',
    _generations_output,
    replace=True,
    use_cache=True,
)
OmegaConf.register_new_resolver(
    'generations_subdir',
    _generations_subdir,
    replace=True,
    use_cache=True,
)


def output_path(name: str) -> Path:
    """Return a path inside the active Hydra invocation."""
    path = Path(HydraConfig.get().runtime.output_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def start_stage(config: DictConfig) -> None:
    """Reserve the output directory and persist the resolved configuration."""
    with output_path('invocation.json').open('x') as file:
        revision = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=False
        ).stdout.strip()
        dirty = subprocess.run(
            ['git', 'status', '--porcelain'], capture_output=True, text=True, check=False
        ).stdout.strip()
        json.dump(
            {
                'revision': revision,
                'dirty': bool(dirty),
                'argv': sys.argv,
                'python': platform.python_version(),
            },
            file,
            indent=2,
        )
    save_config(config)


def save_config(config: DictConfig) -> None:
    """Save all effective settings, including resolved checkpoint-derived values."""
    OmegaConf.save(config=config, f=output_path('config.yaml'), resolve=True)
