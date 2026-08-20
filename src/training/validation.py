import torch
from torch.utils.data import DataLoader

from src.datasets.codec import DatasetCodec
from src.evaluation.scores import AccuracyScores
from src.inference.generate import generate_batch
from src.model import TransformerCVAE
from src.training.loss import LatentKL, Loss, compute_loss


@torch.no_grad()
def validate(
    model: TransformerCVAE,
    loader: DataLoader,
    *,
    kl_weight: float,
    free_bits: float,
    device: torch.device,
) -> tuple[Loss, LatentKL]:
    """
    Run one pass over `loader` without learning from it.
    Args:
        model: The model to evaluate. Put in evaluation mode here, and left in it.
        loader: The dataloader to iterate over. Its batches are `SplitTrace`s.
        kl_weight: The weight this step's KL term is given.
        free_bits: Nats per latent dimension the KL is not penalized below.
        device: The device to run the computations on.
    Returns:
        The metrics of the pass and its KL broken down by latent dimension, both averaged over
        the traces of the split.
    """
    model.eval()

    totals = Loss()
    latent_totals = LatentKL()
    for batch in loader:
        batch = batch.to(device)
        output = model(batch)
        _, metrics, latent_kl = compute_loss(
            output,
            batch,
            pad_activity_index=model.pad_activity_index,
            kl_weight=kl_weight,
            free_bits=free_bits,
        )
        totals += metrics
        latent_totals += latent_kl

    traces = len(loader.dataset)
    return totals / traces, latent_totals / traces


@torch.no_grad()
def validate_generation(
    model: TransformerCVAE,
    loader: DataLoader,
    *,
    num_samples: int,
    codec: DatasetCodec,
    device: torch.device,
) -> AccuracyScores:
    """
    Generate suffixes from the prefixes in `loader` and compare them to the ground truth.

    Scored through `AccuracyScores.of`, the same way the final report is built, over the same
    population: every prefix counts here and in `pipelines/evaluate.py` alike, so a training
    curve and a final report differ only in which split and how much of it they read.

    Args:
        model: The model to evaluate. Put in evaluation mode here, and left in it.
        loader: The prefixes to generate for, from a `TraceDataset`.
        num_samples: Suffixes to draw per prefix. The spread across them is what
            `sample_diversity` measures, and `generate` puts `len(batch) * num_samples` rows
            through the decoder at once, so it is also what the caller sizes its batches by.
        codec: The codec the split was encoded through, read here to put the
            generations back into the log's own units. Passed rather than read off
            `loader.dataset`, which is a `Subset` wherever the split is bigger than the slice
            validated on.
        device: The device to run the computations on.
    Returns:
        The metrics of the pass, averaged over prefixes.
    """
    model.eval()

    scores = [
        AccuracyScores.of(generation)
        for batch in loader
        for generation in generate_batch(
            model=model,
            batch=batch.to(device),
            num_samples=num_samples,
            codec=codec,
        )
    ]
    return AccuracyScores.mean(scores)
