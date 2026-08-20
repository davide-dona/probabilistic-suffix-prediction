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

# Variance of a dimension's displacement above which it counts as used. Burda et al.'s
# active-units threshold, read off the displacement rather than off the posterior mean itself:
# this model's prior is conditioned on the prefix, so under a total collapse the posterior mean
# still moves from prefix to prefix and the usual criterion would call every dimension used.
ACTIVE_USAGE_VARIANCE = 0.01


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
class LatentUsage:
    """How much of z one pass used, dimension by dimension.

    Two quantities, because neither alone is enough. The KL says how far the posterior stands from
    the prior, but a dimension can stand off it on variance alone, with the two means on top of
    each other, and carry nothing about the suffix; free bits makes exactly that shape, since below
    the floor nothing pulls a dimension either way. The displacement, how far the posterior's mean
    sits from the prior's in prior standard deviations, is what the decoder can read instead, and
    its variance across traces is what is left of it once a shift that never changes is discounted.

    Every field is `[latent_dim]` and summed over the traces of the pass, like `Loss`, so batches
    are accumulated with `+` and divided once by the traces they covered. After that division
    `displacement` and `displacement_squares` are the mean and the mean square, which is how
    `variance` reads them. It is logged rather than reported, so, like `Loss`, it declares no unit;
    unlike `Loss` it holds vectors, which `ScalarMetrics` is float-only for.
    """

    # A 0-d zero is an empty total: it broadcasts against the first vector added to it, so an
    # accumulator needs to know nothing of the latent's width, and torch adds a 0-d CPU tensor to
    # a tensor on any device.
    kl_per_dimension: torch.Tensor = torch.zeros(())
    displacement: torch.Tensor = torch.zeros(())
    displacement_squares: torch.Tensor = torch.zeros(())

    @property
    def variance(self) -> torch.Tensor:
        """`[latent_dim]`, how much each dimension's displacement varies from trace to trace.

        Only meaningful once the pass has been divided by the traces it covered, since it reads
        the two displacement fields as a mean and a mean square.
        """
        return self.displacement_squares - self.displacement**2

    def __add__(self, other: 'LatentUsage') -> 'LatentUsage':
        return LatentUsage(
            kl_per_dimension=self.kl_per_dimension + other.kl_per_dimension,
            displacement=self.displacement + other.displacement,
            displacement_squares=self.displacement_squares + other.displacement_squares,
        )

    def __truediv__(self, divisor: float) -> 'LatentUsage':
        return LatentUsage(
            kl_per_dimension=self.kl_per_dimension / divisor,
            displacement=self.displacement / divisor,
            displacement_squares=self.displacement_squares / divisor,
        )

    def log(self, writer: 'SummaryWriter', step: int, *, prefix: str, free_bits: float) -> None:
        """
        Write what the pass put into z, and how many dimensions carried any of it.

        Read of a pass already divided by the traces it covered, since `free_bits` floors a
        per-trace KL and a total would stand above any floor.

        Args:
            writer: The TensorBoard writer to log to.
            step: The step these metrics belong to.
            prefix: Namespace to log under, e.g. `val`, matching the one `Loss` is logged under.
            free_bits: The floor the pass was trained under. What sits below it was never charged
                for, so only what stands above it is KL the model paid to keep.
        """
        variance = self.variance
        writer.add_scalar(
            f'{prefix}/kl_above_floor',
            (self.kl_per_dimension - free_bits).clamp(min=0.0).sum().item(),
            step,
        )
        writer.add_scalar(f'{prefix}/latent_usage', variance.sum().item(), step)
        writer.add_scalar(
            f'{prefix}/active_dimensions',
            (variance > ACTIVE_USAGE_VARIANCE).sum().item(),
            step,
        )


def compute_loss(
    output: TransformerCVAEOutput,
    batch: SplitTrace,
    *,
    pad_activity_index: int,
    kl_weight: float,
    free_bits: float,
) -> tuple[torch.Tensor, Loss, LatentUsage]:
    """Score a forward pass's predictions against the batch it was run on.
    Args:
        output: The model's prediction for `batch`, from `model(batch)`.
        batch: A batch from `TraceDataset`, already on the right device.
        pad_activity_index: `model.pad_activity_index`, ignored in the activity cross-entropy.
        kl_weight: The weight the KL term is given at this step (see `training/kl.py`).
        free_bits: Nats per latent dimension the KL is not penalized below (see `free_bits_kl`).
    Returns:
        The per-trace loss to backpropagate, the metrics to log, and what the pass put into z
        broken down by latent dimension, the last two summed over the batch.
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

    # How far the posterior's mean stands from the prior's, in prior standard deviations. Read
    # only by the diagnostic, so it is built off the graph.
    with torch.no_grad():
        displacement = (output.posterior.mean - output.prior.mean) / torch.exp(
            0.5 * output.prior.logvar
        )  # [batch_size, latent_dim]

    metrics = Loss(
        loss=total_loss.item(),
        reconstruction_loss=reconstruction_loss.item(),
        kl_loss=kl_loss.item(),
        penalized_kl_loss=penalized_kl_loss.item(),
        activity_loss=activity_loss.item(),
        remaining_time_loss=remaining_time_loss.item(),
    )
    latent_usage = LatentUsage(
        kl_per_dimension=kl_per_dim.sum(dim=0).detach(),  # [latent_dim]
        displacement=displacement.sum(dim=0),  # [latent_dim]
        displacement_squares=displacement.pow(2).sum(dim=0),  # [latent_dim]
    )
    return total_loss / batch_size, metrics, latent_usage
