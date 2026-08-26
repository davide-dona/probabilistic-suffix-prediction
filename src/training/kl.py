from dataclasses import dataclass
from typing import Self

import torch

from src.distributions.gaussian import Gaussian
from src.scalar_metrics import ScalarMetrics


def gaussian_kl(posterior: Gaussian, prior: Gaussian) -> torch.Tensor:
    """Closed-form KL divergence between two diagonal Gaussians, per latent dimension.
    Args:
        posterior: The posterior distribution.
        prior: The prior distribution.
    Returns:
        `[batch_size, latent_dim]`, the divergence contributed by each dimension.
    """
    divergence = (
        prior.logvar
        - posterior.logvar
        + (posterior.logvar.exp() + (posterior.mean - prior.mean) ** 2) / prior.logvar.exp()
        - 1.0
    )
    return 0.5 * divergence  # [batch_size, latent_dim]


# Cyclical annealing: https://github.com/haofuml/cyclical_annealing
def cyclical_linear_weight(
    step: int,
    *,
    period_steps: int,
    ratio: float,
    start: float,
    stop: float,
) -> float:
    """
    The KL weight one optimizer step is given under a cyclical annealing schedule.

    Every `period_steps` steps the weight ramps linearly from `start` to `stop` over the first
    `ratio` of the cycle, then holds at the top for the rest of it. This way a run repeatedly gives
    the posterior a stretch of cheap latent capacity and then pays for it again

    Args:
        step: The step to weight, counted from 0.
        period_steps: Length of one cycle, in optimizer steps.
        ratio: Fraction of each cycle spent ramping up, the rest being held at the top.
        start: Weight every cycle ramps up from.
        stop: Weight every cycle ramps up to, and is held at for the rest of the cycle.
    Returns:
        The weight to apply at `step`.
    """
    ramp_steps = period_steps * ratio
    position = step % period_steps  # how far into its own cycle this step is

    # If we already passed the ramp, hold at the top of the cycle
    if position >= ramp_steps:
        return stop
    # Otherwise, linearly interpolate between the start and stop of the cycle
    return start + (stop - start) * (position / ramp_steps)


def free_bits_kl(kl_per_dim: torch.Tensor, *, free_bits: float) -> torch.Tensor:
    """Floor each latent dimension's KL at `free_bits` nats before it is weighted.
    This is a common trick to prevent posterior collapse, where the KL is driven to 0 and the latent
    is ignored.
    Args:
        kl_per_dim: `[batch_size, latent_dim]`, the per-dimension KL from `gaussian_kl`.
        free_bits: Nats per dimension the KL is not penalized below. 0.0 leaves it unfloored.
    Returns:
        A scalar, the batch-summed, dimension-floored KL to weight and backpropagate.
    """
    batch_size = kl_per_dim.size(0)
    return kl_per_dim.mean(dim=0).clamp(min=free_bits).sum() * batch_size


# Margin over `free_bits` a dimension has to clear to count as used: one parked on the floor pays
# no gradient and drifts around it, so counting from the floor exactly would flicker on noise.
ACTIVE_MARGIN_NATS = 0.05


@dataclass(frozen=True, slots=True)
class LatentMetrics(ScalarMetrics):
    """What z carried over one pass: the KL it holds, and how many dimensions hold it.

    Neither is a term of the loss. A total KL cannot tell a collapsed latent from a few dead
    dimensions among used ones, which is what `free_bits` hides: a dimension below the floor is
    left unpenalized, so a run can pay `latent_dim * free_bits` nats and carry no information.
    Logged rather than reported, so like `Loss` it declares no units.
    """

    kl_nats: float = 0.0
    active_dims: float = 0.0

    @classmethod
    def of(cls, kl_per_dim: torch.Tensor, *, free_bits: float) -> Self:
        """Read one batch's KL, counting dimensions on the scale `free_bits_kl` floors.

        Args:
            kl_per_dim: `[batch_size, latent_dim]`, the per-dimension KL from `gaussian_kl`.
            free_bits: Nats per dimension the KL is not penalized below.
        Returns:
            The metrics, scaled by the batch like the loss terms they are logged beside, so the
            caller's division by the batch recovers both.
        """
        batch_size = kl_per_dim.size(0)
        active = (kl_per_dim.mean(dim=0) > free_bits + ACTIVE_MARGIN_NATS).sum().item()
        return cls(kl_nats=kl_per_dim.sum().item(), active_dims=active * batch_size)
