from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from src.datasets.codec import DatasetCodec
from src.evaluation.scores import AccuracyScores, DistributionScores
from src.inference.generate import generate_batch
from src.logs.continuations import ContinuationIndex
from src.model import TransformerCVAE
from src.suffixes import ActivityCodes
from src.training.kl import LatentMetrics
from src.training.loss import Loss, compute_loss


@dataclass(frozen=True, slots=True)
class GenerationMetrics:
    """What one generation pass measured: how close the suffixes are to the ground truth, and how
    the set of them compares against every continuation the validation split took.

    The two families a training run reads. Accuracy is the curve a run is watched on, and
    `DistributionScores.emsc` is the score a checkpoint is selected on.
    """

    accuracy: AccuracyScores
    distribution: DistributionScores


@torch.no_grad()
def validate(
    model: TransformerCVAE,
    loader: DataLoader,
    *,
    kl_weight: float,
    free_bits: float,
    device: torch.device,
) -> tuple[Loss, LatentMetrics]:
    """
    Run one pass over `loader` without learning from it.
    Args:
        model: The model to evaluate. Put in evaluation mode here, and left in it.
        loader: The dataloader to iterate over. Its batches are `SplitTrace`s.
        kl_weight: The weight this step's KL term is given.
        free_bits: Nats per latent dimension the KL is not penalized below.
        device: The device to run the computations on.
    Returns:
        The loss terms of the pass and what the latent carried, both averaged over the traces
        of the split.
    """
    model.eval()

    totals = Loss()
    latent_totals = LatentMetrics()
    for batch in loader:
        batch = batch.to(device)
        output = model(batch)
        _, metrics, latent = compute_loss(
            output,
            batch,
            pad_activity_index=model.pad_activity_index,
            kl_weight=kl_weight,
            free_bits=free_bits,
        )
        totals += metrics
        latent_totals += latent

    traces = len(loader.dataset)
    return totals / traces, latent_totals / traces


@torch.no_grad()
def validate_generation(
    model: TransformerCVAE,
    loader: DataLoader,
    *,
    num_samples: int,
    codec: DatasetCodec,
    index: ContinuationIndex,
    device: torch.device,
) -> GenerationMetrics:
    """
    Generate suffixes from the prefixes in `loader` and compare them to the ground truth and to
    every continuation the split was observed to take.

    Scored through the same two families the final report is built from, over the same population:
    every prefix counts here and in `pipelines/evaluate.py` alike. What differs is which split is
    read, how much of it, and how many suffixes each prefix is answered with, so a training curve
    is read for its shape over steps rather than against a report's numbers.

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
        index: The continuations the validation split takes after each of its prefixes. The
            validation split's and never the test split's: selecting a checkpoint against the
            test split's continuations would fold the held-out set into what gets kept.
        device: The device to run the computations on.
    Returns:
        The metrics of the pass, averaged over prefixes.
    """
    model.eval()

    # The same codebook the index was seeded from, so a generated suffix is spelled the way the
    # continuations it is scored against are and nothing is translated per prefix.
    codes = ActivityCodes.of(codec.activity.names)

    generations = [
        generation
        for batch in loader
        for generation in generate_batch(
            model=model,
            batch=batch.to(device),
            num_samples=num_samples,
            codec=codec,
            codes=codes,
        )
    ]
    return GenerationMetrics(
        accuracy=AccuracyScores.mean([AccuracyScores.of(one) for one in generations]),
        distribution=DistributionScores.mean(
            [DistributionScores.of(one, index=index) for one in generations]
        ),
    )
