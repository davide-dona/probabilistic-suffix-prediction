import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import paths
from src.configs import ExperimentConfig, load_config
from src.datasets.codec import DatasetCodec
from src.datasets.dataset import TraceDataset
from src.identity import RunIdentity, require_same_dataset
from src.inference import (
    generate_batch,
    generation_batch_size,
    open_generations,
    table_from_generations,
)
from src.model import TransformerCVAE, load_checkpoint
from src.paths import Split


def run(config: ExperimentConfig, model_path: Path) -> None:
    """Generate suffixes for every prefix of the test split and write them out.

    Args:
        config: The validated experiment config.
        model_path: The checkpoint to generate with. Named rather than guessed at: a config
            matches every run ever started from it, and picking one of them is a decision the
            caller makes, not one to be inferred from a filename.
    Raises:
        ValueError: If the checkpoint was trained on a different dataset than the config
            describes, which would decode its output through the wrong codec.
    """
    paths.require_dataset(config.data.name)
    torch.manual_seed(config.seed)

    print('Loading codec...', flush=True)
    codec = DatasetCodec.load(config.data)

    # Load the model from the checkpoint and put it on the right device
    checkpoint = load_checkpoint(model_path)
    run = RunIdentity.from_dict(checkpoint['run'])
    require_same_dataset(run, config.data.name, artifact=model_path)

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
        f'{len(test_dataset)} test prefixes, with {model_path}',
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
        '-c',
        '--config',
        required=True,
        help="Name of this experiment's dataset config, from config/datasets/ (e.g. 'bpic17').",
    )
    parser.add_argument(
        '-w',
        '--hardware',
        required=True,
        help='Hardware profile to run under, from config/hardware/.',
    )
    parser.add_argument(
        '-m',
        '--model',
        type=Path,
        required=True,
        help='Path to the checkpoint to generate with, from '
        '`best-models/` or `outputs/checkpoints/`.',
    )
    args = parser.parse_args()

    run(load_config(args.config, args.hardware), args.model)


if __name__ == '__main__':
    main()
