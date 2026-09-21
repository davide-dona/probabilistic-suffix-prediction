from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import torch
import wandb
from torch.utils.data import DataLoader

from src.datasets.codec import DatasetCodec
from src.evaluation.scores import (
    ActivityDiagnostics,
    ActivityScores,
    ConformanceDiagnostics,
    ConformanceScores,
    SuffixLengthDiagnostics,
    SuffixLengthScores,
    TimeDiagnostics,
)
from src.evaluation.summary import PrefixSummary
from src.inference.generate import generate_batch
from src.logs.declare import ConformanceChecker
from src.suffixes import ActivityCodes
from src.training.kl import LatentMetrics
from src.training.loss import Loss
from src.visualization.catalogue import TABLES

if TYPE_CHECKING:
    from src.model import SuffixModel

_TABLE_OF_METRIC = {entry.key: table.name for table in TABLES for entry in table.columns}
_DIAGNOSTICS = 'diagnostics'


@dataclass(frozen=True, slots=True)
class GenerationMetrics:
    """Validation metrics, with energy score averaged over every generated example."""

    activity: ActivityScores
    suffix_length: SuffixLengthScores
    conformance: ConformanceScores
    activity_diagnostics: ActivityDiagnostics
    suffix_length_diagnostics: SuffixLengthDiagnostics
    time_diagnostics: TimeDiagnostics
    conformance_diagnostics: ConformanceDiagnostics

    def log(self, step: int) -> None:
        """Log scores by table and diagnostics by source field.

        Args:
            step: The training step this pass scores.
        """
        families = (
            (self.activity, None),
            (self.suffix_length, None),
            (self.conformance, 'conformance'),
        )
        payload = {}
        for family, table_namespace in families:
            for declaration in type(family).metrics():
                namespace = table_namespace or _TABLE_OF_METRIC.get(declaration.key, _DIAGNOSTICS)
                payload[f'{namespace}/{declaration.key}'] = getattr(family, declaration.key)
        for field, diagnostics in (
            ('activity', self.activity_diagnostics),
            ('suffix-length', self.suffix_length_diagnostics),
            ('time', self.time_diagnostics),
            ('conformance', self.conformance_diagnostics),
        ):
            payload.update(
                {
                    f'{_DIAGNOSTICS}/{field}/{name}': value
                    for name, value in asdict(diagnostics).items()
                }
            )
        wandb.log(payload, step=step)


@torch.no_grad()
def validate(
    model: SuffixModel, loader: DataLoader, *, step: int, device: torch.device
) -> tuple[Loss, LatentMetrics | None]:
    """
    Run one pass over `loader` without learning from it.
    Args:
        model: The model to evaluate. Put in evaluation mode here, and left in it.
        loader: The dataloader to iterate over. Its batches are `SplitTrace`s.
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
    return GenerationMetrics(
        activity=ActivityScores.mean([summary.activity for summary in summaries]),
        suffix_length=SuffixLengthScores.mean([summary.suffix_length for summary in summaries]),
        conformance=ConformanceScores.mean([summary.conformance for summary in summaries]),
        activity_diagnostics=ActivityDiagnostics.mean(
            [summary.activity_diagnostics for summary in summaries]
        ),
        suffix_length_diagnostics=SuffixLengthDiagnostics.mean(
            [summary.suffix_length_diagnostics for summary in summaries]
        ),
        time_diagnostics=TimeDiagnostics.mean([summary.time_diagnostics for summary in summaries]),
        conformance_diagnostics=ConformanceDiagnostics.mean(
            [summary.conformance_diagnostics for summary in summaries]
        ),
    )
