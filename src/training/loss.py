from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.datasets.dataset import SplitTrace
from src.model import TransformerCVAEOutput
from src.scalar_metrics import ScalarMetrics
from src.training.kl import LatentMetrics, free_bits_kl, gaussian_kl


@dataclass(frozen=True)
class Loss(ScalarMetrics):
    """The loss of one pass, and the terms it is made of."""

    loss: float = 0.0
    reconstruction_loss: float = 0.0
    # The KL after each dimension is floored at `free_bits`, which is what `loss` charges for.
    # The weight it is charged at is `kl_weight`, logged on its own.
    floored_kl_loss: float = 0.0
    activity_loss: float = 0.0
    activity_duration_loss: float = 0.0
    remaining_time_loss: float = 0.0


def compute_loss(
    output: TransformerCVAEOutput,
    batch: SplitTrace,
    *,
    pad_activity_index: int,
    kl_weight: float,
    free_bits: float,
) -> tuple[torch.Tensor, Loss, LatentMetrics]:
    """Score a forward pass's predictions against the batch it was run on.
    Args:
        output: The model's prediction for `batch`, from `model(batch)`.
        batch: A batch from `TraceDataset`, already on the right device.
        pad_activity_index: `model.pad_activity_index`, ignored in the activity cross-entropy.
        kl_weight: The weight the KL term is given at this step (see `training/kl.py`).
        free_bits: Nats per latent dimension the KL is not penalized below (see `free_bits_kl`).
    Returns:
        The per-trace loss to backpropagate, the terms it is made of, and what the latent
        carried, the last two summed over the batch.
    """
    batch_size = batch.suffix.activities.size(0)

    # Compute the activity reconstruction loss, summed over the batch. The EOT closing a suffix
    # is one of the positions scored here: stopping is something the decoder has to predict.
    activity_loss = F.cross_entropy(
        output.decoder.activity_logits.transpose(1, 2),
        batch.suffix.activities,
        ignore_index=pad_activity_index,
        reduction='sum',
    )

    # Both time targets are defined at the suffix's content alone: the EOT has no time of its
    # own, and `suffix.length` counts it.
    timed = _timed_positions(batch)  # [batch_size, seq_len]
    activity_duration_loss = _squared_error(
        prediction=output.decoder.activity_durations, target=batch.activity_durations, mask=timed
    )
    remaining_time_loss = _squared_error(
        prediction=output.decoder.remaining_times, target=batch.remaining_times, mask=timed
    )

    reconstruction_loss = activity_loss + activity_duration_loss + remaining_time_loss

    # Compute KL divergence, summed over the batch
    kl_per_dim = gaussian_kl(
        posterior=output.posterior, prior=output.prior
    )  # [batch_size, latent_dim]
    floored_kl_loss = free_bits_kl(kl_per_dim, free_bits=free_bits)
    total_loss = reconstruction_loss + kl_weight * floored_kl_loss

    metrics = Loss(
        loss=total_loss.item(),
        reconstruction_loss=reconstruction_loss.item(),
        floored_kl_loss=floored_kl_loss.item(),
        activity_loss=activity_loss.item(),
        activity_duration_loss=activity_duration_loss.item(),
        remaining_time_loss=remaining_time_loss.item(),
    )
    latent = LatentMetrics.of(kl_per_dim, free_bits=free_bits)
    return total_loss / batch_size, metrics, latent


def _timed_positions(batch: SplitTrace) -> torch.Tensor:
    """Mark the suffix positions the time targets are defined at.

    Args:
        batch: A batch from `TraceDataset`, whose `suffix.length` counts the EOT closing it.
    Returns:
        `[batch_size, seq_len]`, True at the positions holding a real event.
    """
    positions = torch.arange(
        end=batch.suffix.activities.size(dim=1), device=batch.suffix.length.device
    )  # [seq_len]
    return positions.unsqueeze(dim=0) < (batch.suffix.length - 1).unsqueeze(dim=1)


def _squared_error(
    *, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Half the squared error of one point head, over the positions its target is defined at.

    A head emitting its own scale could buy its way out of a hard position by widening; at a
    fixed unit scale the term is floored at 0 and has nothing to shrink.

    Args:
        prediction: The head's output, `[batch_size, seq_len]`, standardized.
        target: What it is scored against, the same shape and scale.
        mask: True at the positions to score.
    Returns:
        A scalar, summed over the scored positions of the whole batch.
    """
    error = 0.5 * (prediction - target).pow(exponent=2)  # [batch_size, seq_len]
    return error.masked_fill(mask=~mask, value=0.0).sum()
