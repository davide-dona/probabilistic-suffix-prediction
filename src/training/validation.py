from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import wandb
from torch.utils.data import DataLoader

from src.activity_codes import ActivityCodes
from src.datasets.codec import DatasetCodec
from src.evaluation.scores import METRICS
from src.evaluation.summary import PrefixSummary, ScoreGroups
from src.inference.generate import generate_batch
from src.logs.declare import ConformanceChecker
from src.training.kl import LatentMetrics
from src.training.loss import Loss

if TYPE_CHECKING:
    from src.model import SuffixModel

ACTIVITY_LOG_NAMESPACE = 'generation/activity'


@dataclass(frozen=True, slots=True)
class GenerationMetrics:
    """Validation metrics, with energy score averaged over every generated example."""

    scores: ScoreGroups

    def log(self, step: int) -> None:
        """Log every registered value under its declared evaluation group.

        Args:
            step: The training step this pass scores.
        """
        namespaces = {
            'activity': ACTIVITY_LOG_NAMESPACE,
            'suffix_length': 'generation/suffix-length',
            'time': 'generation/time',
            'conformance': 'generation/conformance',
        }
        values = self.scores.flatten()
        wandb.log(
            {
                f'{namespaces[metric.group]}/{key}': values[key]
                for key, metric in METRICS.entries.items()
            },
            step=step,
        )


@torch.no_grad()
def validate(
    model: SuffixModel, loader: DataLoader, *, step: int, device: torch.device
) -> tuple[Loss, LatentMetrics | None]:
    """
    Run one pass over `loader` without learning from it.
    Args:
        model: The model to evaluate. Put in evaluation mode here, and left in it.
        loader: The dataloader to iterate over. Its batches are `TraceCut`s.
        step: The training step this pass scores, for a model whose loss anneals a term over
            the run.
        device: The device to run the computations on.
    Returns:
        The loss terms of the pass and what the latent carried, both averaged over the traces
        of the split. The latter is None where the model has no latent.
    """
    model.eval()

    totals = Loss()
    latent_totals: LatentMetrics | None = None
    for batch in loader:
        batch = batch.to(device)
        output = model(batch)
        _, metrics, latent = model.compute_loss(output, batch, step=step)
        totals += metrics
        if latent is not None:
            latent_totals = latent if latent_totals is None else latent_totals + latent

    traces = len(loader.dataset)
    return totals / traces, None if latent_totals is None else latent_totals / traces


@torch.no_grad()
def validate_generation(
    model: SuffixModel,
    loader: DataLoader,
    *,
    num_samples: int,
    codec: DatasetCodec,
    checker: ConformanceChecker,
    device: torch.device,
) -> GenerationMetrics:
    """
    Generate suffixes from the prefixes in `loader` and compare them to the ground truth and the
    declarative model.

    Scored through the same families the final report is built from, and each prefix is answered
    with the same number of suffixes. What differs is which split is read and how much of it, so a
    training curve sits on a report's scale without being a report's number.

    Energy score is measured against each example's observed suffix.

    Args:
        model: The model to evaluate. Put in evaluation mode here, and left in it.
        loader: The prefixes to generate for, from a `TraceDataset`.
        num_samples: Suffixes to draw per prefix. `generate` puts `len(batch) * num_samples` rows
            through the decoder at once, so it is also what the caller sizes its batches by.
        codec: The codec the split was encoded through, read here to put the generations back into
            the log's own units. Passed rather than read off
            `loader.dataset`, which is a `Subset` wherever the split is bigger than the slice
            validated on.
        checker: The declarative model to check generated suffixes against.
        device: The device to run the computations on.
    Returns:
        The metrics of the pass, averaged over prefixes.
    """
    model.eval()

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
    if not generations:
        raise ValueError('Validation generation subset is empty')
    summaries = [PrefixSummary.of(one, checker=checker) for one in generations]
    return GenerationMetrics(scores=ScoreGroups.mean([summary.scores for summary in summaries]))
