import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import paths
from src.cli import add_hardware_argument, existing_file
from src.configs import load_generation_config
from src.datasets.codec import DatasetCodec
from src.datasets.dataset import TraceDataset
from src.identity import RunIdentity
from src.inference.generate import generate_batch, generation_batch_size
from src.inference.generation_store import open_generations, table_from_generations
from src.model import TransformerCVAE, load_checkpoint
from src.paths import Split


def run(checkpoint_path: Path, *, hardware: Path, num_samples: int | None) -> None:
    """Generate suffixes for every prefix of the test split and write them out.

    Args:
        checkpoint_path: The checkpoint to generate with. Named rather than guessed at: a config
            matches every run ever started from it, and picking one of them is a decision the
            caller makes, not one to be inferred from a filename. It carries the config of the
            run that wrote it, so nothing about the model or the dataset is passed alongside it.
        hardware: The hardware profile to generate under, replacing the one the run was trained
            with.
        num_samples: How many suffixes to draw per prefix, or `None` for the run's own.
    """
    print(f'Loading the checkpoint at {checkpoint_path}...', flush=True)
    checkpoint = load_checkpoint(checkpoint_path)
    run = RunIdentity.from_dict(checkpoint['run'])
    config = load_generation_config(
        checkpoint['experiment_config'], hardware=hardware, num_samples=num_samples
    )

    paths.require_dataset(config.data.name)
    torch.manual_seed(config.seed)

    print('Loading codec...', flush=True)
    codec = DatasetCodec.load(config.data)

    model = TransformerCVAE.from_checkpoint(checkpoint, codec, device=config.training.device)
    model.eval()

    # Build the DataLoader for the test split
    print('Loading test split...', flush=True)
    test_dataset = TraceDataset(codec=codec, split=Split.TEST)

    print('Sorting prefixes by suffix length...', flush=True)
    sampler = test_dataset.length_sorted_indices()

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=generation_batch_size(
            inference=config.inference, prefixes_upper_bound=config.dataloader.batch_size
        ),
        # sort the prefixes by length so the batches are more uniform and generation is faster
        sampler=sampler,
        num_workers=config.dataloader.num_workers,
    )

    # The output file is named after the run the checkpoint carries, not after the file it was
    # read from, so the generations land under the run that produced them whatever it is called.
    path = paths.generations_path(run)
    device = torch.device(config.training.device)

    print(
        f'Generating {config.inference.num_samples} suffixes for each of '
        f'{len(test_dataset)} test prefixes, with {checkpoint_path}',
        flush=True,
    )

    # Write the generation while it is being produced, avoiding a huge in-memory DataFrame.
    with open_generations(path, run) as parquet_writer:
        for batch in tqdm(iterable=test_loader, desc='Generating', unit='batch'):
            generations = generate_batch(
                model=model,
                batch=batch.to(device),
                num_samples=config.inference.num_samples,
                codec=codec,
            )
            # Write the generations to the Parquet file in a single table, one row per prefix.
            parquet_writer.write_table(table=table_from_generations(generations))

    print(f'Wrote generated suffixes to {path}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate test-split suffixes from a trained model.'
    )
    parser.add_argument(
        '-m',
        '--checkpoint',
        type=existing_file,
        metavar='CHECKPOINT',
        required=True,
        help='Path to the checkpoint to generate with, from `pretrained/`, '
        '`outputs/checkpoints/best/` or `outputs/checkpoints/last/`. Its own config is what '
        'the model and the dataset are read from.',
    )
    add_hardware_argument(parser, required=True)
    parser.add_argument(
        '-n',
        '--num-samples',
        type=int,
        default=None,
        metavar='N',
        help='How many suffixes to draw per prefix, overriding the config the checkpoint '
        "carries. Defaults to the run's own `inference.num_samples`.",
    )
    args = parser.parse_args()

    run(args.checkpoint, hardware=args.hardware, num_samples=args.num_samples)


if __name__ == '__main__':
    main()
