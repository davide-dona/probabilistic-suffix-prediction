from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.datasets.dataset import SplitTrace
from src.distributions.gaussian import Gaussian
from src.model import ModelOutput
from src.scalar_metrics import ScalarMetrics
from src.training.kl import LatentMetrics, free_bits_kl, gaussian_kl


@dataclass(frozen=True)
class Loss(ScalarMetrics):
    """The loss of one pass, and the terms it is made of.

    One shape for both architectures, so a curve reads the same whichever produced it. A model
    with no latent leaves `floored_kl_loss` at 0.0 and its `loss` is the reconstruction alone.
    """

    loss: float = 0.0
    reconstruction_loss: float = 0.0
    # The KL after each dimension is floored at `free_bits`, which is what `loss` charges for.
    # The weight it is charged at is `kl_weight`, logged on its own.
    floored_kl_loss: float = 0.0
    activity_loss: float = 0.0
    time_to_next_loss: float = 0.0
    remaining_time_loss: float = 0.0


def compute_loss(
    output: ModelOutput,
    batch: SplitTrace,
    *,
    pad_activity_index: int,
    kl_weight: float,
    free_bits: float,
) -> tuple[torch.Tensor, Loss, LatentMetrics | None]:
    """Score a forward pass's predictions against the batch it was run on.

    Whether the pass is scored by the ELBO or by its reconstruction alone is read off the output
    rather than passed in: a model with no latent has no KL term to charge and nothing for
    `kl_weight` or `free_bits` to weigh.

    Args:
        output: The model's prediction for `batch`, from `model(batch)`.
        batch: A batch from `TraceDataset`, already on the right device.
        pad_activity_index: `model.pad_activity_index`, ignored in the activity cross-entropy.
        kl_weight: The weight the KL term is given at this step (see `training/kl.py`).
        free_bits: Nats per latent dimension the KL is not penalized below (see `free_bits_kl`).
    Returns:
        The per-trace loss to backpropagate, the terms it is made of, and what the latent
        carried, the last two summed over the batch. The last is None where the model has no
        latent for the watchdogs to read.
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
    time_to_next_loss = _gaussian_nll(
        prediction=output.decoder.times_to_next, target=batch.times_to_next, mask=timed
    )
    remaining_time_loss = _gaussian_nll(
        prediction=output.decoder.remaining_times, target=batch.remaining_times, mask=timed
    )

    reconstruction_loss = activity_loss + time_to_next_loss + remaining_time_loss

    if output.latents is None:
        metrics = Loss(
            loss=reconstruction_loss.item(),
            reconstruction_loss=reconstruction_loss.item(),
            activity_loss=activity_loss.item(),
            time_to_next_loss=time_to_next_loss.item(),
            remaining_time_loss=remaining_time_loss.item(),
        )
        return reconstruction_loss / batch_size, metrics, None

    # Compute KL divergence, summed over the batch
    kl_per_dim = gaussian_kl(
        posterior=output.latents.posterior, prior=output.latents.prior
    )  # [batch_size, latent_dim]
    floored_kl_loss = free_bits_kl(kl_per_dim, free_bits=free_bits)
    total_loss = reconstruction_loss + kl_weight * floored_kl_loss

    metrics = Loss(
        loss=total_loss.item(),
        reconstruction_loss=reconstruction_loss.item(),
        floored_kl_loss=floored_kl_loss.item(),
        activity_loss=activity_loss.item(),
        time_to_next_loss=time_to_next_loss.item(),
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


def _gaussian_nll(
    *, prediction: Gaussian, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """The negative log-likelihood of one time head, over the positions its target is defined at,
    without the constant the two halves of a comparison would both carry.

    A head that emits its own scale can buy its way out of a hard position by widening, which is
    the price of letting it be sampled from: a model whose draws come from its heads has to say
    how wide they are. A head pinned to unit variance pays nothing for the first term and this
    reduces exactly to the half-squared error, which is floored at 0 and has nothing to shrink.

    Args:
        prediction: The head's distribution, `[batch_size, seq_len]` per field, standardized.
        target: What it is scored against, the shape and scale of `prediction.mean`.
        mask: True at the positions to score.
    Returns:
        A scalar, summed over the scored positions of the whole batch.
    """
    # exp(-logvar) rather than a division by the variance: one exp and a multiply, and no
    # reciprocal of a number that a wide head can drive towards 0.
    error = 0.5 * (
        prediction.logvar
        + (prediction.mean - target).pow(exponent=2) * torch.exp(input=-prediction.logvar)
    )  # [batch_size, seq_len]
    return error.masked_fill(mask=~mask, value=0.0).sum()
