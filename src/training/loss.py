from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from src.datasets.dataset import SplitTrace
from src.model import TransformerCVAEOutput
from src.scalar_metrics import ScalarMetrics
from src.training.kl import free_bits_kl, gaussian_kl

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

# Nats above which a latent dimension counts as used where no free-bits floor is set. A dimension
# below the floor gets no KL gradient at all, so wherever one is set the floor is what use means.
ACTIVE_KL_NATS = 0.01


@dataclass(frozen=True)
class Loss(ScalarMetrics):
    """The loss of one pass, and the terms it is made of."""

    loss: float = 0.0
    reconstruction_loss: float = 0.0
    kl_loss: float = 0.0
    penalized_kl_loss: float = 0.0
    activity_loss: float = 0.0
    remaining_time_loss: float = 0.0


@dataclass(frozen=True, slots=True)
class LatentKL:
    """One pass's KL, dimension by dimension: what tells a collapsed latent apart from a few dead
    dimensions among used ones.

    The sum alone cannot: 16 dimensions parked on a 0.3-nat floor and three dimensions carrying
    everything while thirteen do nothing add up to much the same number.

    Summed over the traces of the pass, like `Loss`, so batches are accumulated with `+` and
    divided once by the traces they covered. It is logged rather than reported, so, like `Loss`,
    it declares no unit; unlike `Loss` it holds a vector, which `ScalarMetrics` is float-only for.
    """

    # [latent_dim]. A 0-d zero is an empty total: it broadcasts against the first vector added to
    # it, so an accumulator needs to know nothing of the latent's width, and torch adds a 0-d CPU
    # tensor to a tensor on any device.
    per_dimension: torch.Tensor = torch.zeros(())

    def __add__(self, other: 'LatentKL') -> 'LatentKL':
        return LatentKL(per_dimension=self.per_dimension + other.per_dimension)

    def __truediv__(self, divisor: float) -> 'LatentKL':
        return LatentKL(per_dimension=self.per_dimension / divisor)

    def log(self, writer: 'SummaryWriter', step: int, *, prefix: str, free_bits: float) -> None:
        """
        Write one curve per latent dimension, and how many of them the pass used.

        Read of a pass already divided by the traces it covered, since `free_bits` floors a
        per-trace KL and a total would stand above any floor.

        Args:
            writer: The TensorBoard writer to log to.
            step: The step these metrics belong to.
            prefix: Namespace to log under, e.g. `val`, matching the one `Loss` is logged under.
            free_bits: The floor the pass was trained under, which a dimension has to stand above
                to be carrying anything the KL term ever asked for.
        """
        threshold = max(free_bits, ACTIVE_KL_NATS)
        per_dimension = self.per_dimension.tolist()
        # Zero-padded, so TensorBoard's own sort lists the dimensions in order under one heading.
        for index, value in enumerate(per_dimension):
            writer.add_scalar(f'{prefix}/kl_per_dimension/{index:02d}', value, step)
        writer.add_scalar(
            f'{prefix}/active_dimensions',
            sum(value > threshold for value in per_dimension),
            step,
        )


def compute_loss(
    output: TransformerCVAEOutput,
    batch: SplitTrace,
    *,
    pad_activity_index: int,
    kl_weight: float,
    free_bits: float,
) -> tuple[torch.Tensor, Loss, LatentKL]:
    """Score a forward pass's predictions against the batch it was run on.
    Args:
        output: The model's prediction for `batch`, from `model(batch)`.
        batch: A batch from `TraceDataset`, already on the right device.
        pad_activity_index: `model.pad_activity_index`, ignored in the activity cross-entropy.
        kl_weight: The weight the KL term is given at this step (see `training/kl.py`).
        free_bits: Nats per latent dimension the KL is not penalized below (see `free_bits_kl`).
    Returns:
        The per-trace loss to backpropagate, the metrics to log, and the same KL broken down by
        latent dimension, the last two summed over the batch.
    """
    batch_size = batch.suffix.activities.size(0)

    # Compute activity and remaining time reconstruction losses, summed over the batch.
    activity_loss = F.cross_entropy(
        output.decoder.activity_logits.transpose(1, 2),
        batch.suffix.activities,
        ignore_index=pad_activity_index,
        reduction='sum',
    )
    remaining_time_loss = output.decoder.remaining_time_distr.nll(batch.remaining_time)
    reconstruction_loss = activity_loss + remaining_time_loss

    # Compute KL divergence, summed over the batch
    kl_per_dim = gaussian_kl(
        posterior=output.posterior, prior=output.prior
    )  # [batch_size, latent_dim]
    kl_loss = kl_per_dim.sum()
    penalized_kl_loss = free_bits_kl(kl_per_dim, free_bits=free_bits)
    total_loss = reconstruction_loss + kl_weight * penalized_kl_loss

    metrics = Loss(
        loss=total_loss.item(),
        reconstruction_loss=reconstruction_loss.item(),
        kl_loss=kl_loss.item(),
        penalized_kl_loss=penalized_kl_loss.item(),
        activity_loss=activity_loss.item(),
        remaining_time_loss=remaining_time_loss.item(),
    )
    latent_kl = LatentKL(per_dimension=kl_per_dim.sum(dim=0).detach())  # [latent_dim]
    return total_loss / batch_size, metrics, latent_kl
