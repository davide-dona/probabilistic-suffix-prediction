import argparse
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src import paths
from src.configs import ExperimentConfig, load_config
from src.datasets.codec import DatasetCodec
from src.datasets.dataset import TraceDataset, fixed_subset
from src.identity import RunIdentity
from src.inference.generate import generation_batch_size
from src.model import TransformerCVAE, load_checkpoint
from src.paths import Split
from src.training.train import train


def resumed(resume_path: Path) -> tuple[ExperimentConfig, dict]:
    """
    Read a checkpoint together with the config of the run that wrote it.

    Args:
        resume_path: The checkpoint to carry on from.
    Returns:
        Its config, and the checkpoint itself.
    Raises:
        ValueError: If the checkpoint carries no training state, and so can only be generated with.
    """
    checkpoint = load_checkpoint(resume_path)
    missing = {
        'experiment_config',
        'run',
        'optimizer_state',
        'early_stopping_state',
        'rng_state',
    } - checkpoint.keys()
    if missing:
        raise ValueError(
            f'{resume_path} is missing {sorted(missing)}: it can be generated with, but not '
            'resumed from. Start a new run instead.'
        )
    return ExperimentConfig.model_validate(checkpoint['experiment_config']), checkpoint


def run(config: ExperimentConfig, checkpoint: dict | None = None) -> None:
    """
    Train the model an experiment config describes, on the dataset it names.
    The dataset must have been preprocessed already.
    Args:
        config: The validated experiment config.
        checkpoint: A checkpoint to carry on from, as read by `resumed`, or `None` to start a
            new run. The run keeps the identity the checkpoint carries, so it writes to the
            TensorBoard directory and the files the interrupted run was writing to.
    """
    paths.require_dataset(config.data.name)

    # Seeded before anything is built, so weight initialization and shuffling are both reproducible.
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)

    print('Loading codec...', flush=True)
    codec = DatasetCodec.load(config.data)

    model = TransformerCVAE(config.model, codec).to(config.training.device)

    # Build the datasets and loaders
    print('Loading train split...', flush=True)
    train_dataset = TraceDataset(codec=codec, split=Split.TRAIN)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.dataloader.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.dataloader.num_workers,
    )

    print('Loading validation split...', flush=True)
    validation_dataset = TraceDataset(codec=codec, split=Split.VAL)
    # Validation and generation loaders are fixed subsets of the validation split, so every run
    # of a config reads the same traces and their curves can be laid over each other.
    val_loader = DataLoader(
        dataset=fixed_subset(
            validation_dataset, size=config.training.validation_pairs, generator=generator
        ),
        batch_size=config.dataloader.batch_size,
        shuffle=False,
        num_workers=config.dataloader.num_workers,
    )
    generation_loader = DataLoader(
        dataset=fixed_subset(
            validation_dataset, size=config.training.generation_pairs, generator=generator
        ),
        batch_size=generation_batch_size(
            inference=config.inference, prefixes_upper_bound=config.dataloader.batch_size
        ),
        shuffle=False,
        num_workers=config.dataloader.num_workers,
    )

    print(
        f'Training on {len(train_loader.dataset)} prefix/suffix pairs, scoring '
        f'{len(val_loader.dataset)} of the {len(validation_dataset)} validation pairs and '
        f'generating for {len(generation_loader.dataset)}'
    )

    # When resuming, keep the identity the checkpoint carries, so it writes to the same
    # TensorBoard directory and the same files.
    run = (
        RunIdentity(
            dataset=config.data.name,
            model=config.model.name,
            tag=f'{datetime.now():%Y%m%d-%H%M%S}',
        )
        if checkpoint is None
        else RunIdentity.from_dict(checkpoint['run'])
    )

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        generation_loader=generation_loader,
        run=run,
        experiment_config=config.model_dump(),
        generator=generator,
        resume=checkpoint,
        generation_samples=config.inference.num_samples,
        codec=codec,
        loss_config=config.loss,
        optimizer_config=config.optimizer,
        training=config.training,
        early_stopping_config=config.early_stopping,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Train a suffix-prediction model.')
    # A run is either started from a config or carried on from a checkpoint, which already
    # describes the run that wrote it. Two descriptions of one run could only disagree.
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        '-c',
        '--config',
        help="Name of this experiment's dataset config, from config/datasets/ (e.g. 'bpic17'), "
        'to start a new run.',
    )
    source.add_argument(
        '-r',
        '--resume',
        type=Path,
        help='Path to a checkpoint to carry on from, its config included. The '
        'run keeps its name, so it writes to the same TensorBoard directory '
        'and the same files.',
    )
    parser.add_argument(
        '-w',
        '--hardware',
        help='Hardware profile to run under, from config/hardware/. Required with '
        '-c/--config; not used with -r/--resume, whose config is already resolved.',
    )
    args = parser.parse_args()

    if args.resume is not None:
        run(*resumed(args.resume))
    else:
        if args.hardware is None:
            parser.error('-w/--hardware is required with -c/--config')
        run(load_config(args.config, args.hardware))


if __name__ == '__main__':
    main()
