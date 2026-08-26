from typing import TYPE_CHECKING

import torch

from src.distributions.gaussian import Gaussian

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


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


# Margin over `free_bits` a dimension has to clear to be counted as used. A dimension parked on
# the floor pays no gradient and drifts around it between passes, so counting it from the floor
# exactly would make the count flicker on noise rather than on the latent changing.
ACTIVE_MARGIN_NATS = 0.05


def active_latent_dims(kl_per_dim: torch.Tensor, *, free_bits: float) -> int:
    """How many latent dimensions carry meaningfully more KL than the free-bits floor.

    A total KL cannot tell a fully collapsed latent from a handful of dead dimensions among used
    ones, which is exactly what `free_bits` hides: a dimension below the floor is left unpenalized,
    so a run can pay `latent_dim * free_bits` nats and still carry no information at all. This
    counts the dimensions the floor is not what is holding up.

    Args:
        kl_per_dim: `[latent_dim]`, the nats each dimension contributed, averaged over the traces
            of the pass, which is the scale `free_bits_kl` floors on.
        free_bits: Nats per dimension the KL is not penalized below, the same value the pass was
            scored with.
    Returns:
        The number of dimensions above `free_bits` by at least `ACTIVE_MARGIN_NATS`.
    """
    return int((kl_per_dim > free_bits + ACTIVE_MARGIN_NATS).sum().item())


def log_kl_diagnostics(
    kl_per_dim: torch.Tensor,
    writer: 'SummaryWriter',
    step: int,
    *,
    free_bits: float,
    prefix: str,
) -> None:
    """Write what each latent dimension carried, and how many of them are used, to TensorBoard.

    The count is the headline, the per-dimension curves what it is read against: once the count
    says three dimensions are alive, only the curves say whether it is the same three across a
    cycle of the annealing schedule.

    Args:
        kl_per_dim: `[latent_dim]`, the nats each dimension contributed, averaged over the traces
            of the pass.
        writer: The TensorBoard writer to log to.
        step: The step these belong to.
        free_bits: Nats per dimension the KL is not penalized below, the same value the pass was
            scored with.
        prefix: Namespace to log under, e.g. `val`, so these sit beside the total KL of the same
            pass.
    """
    writer.add_scalar(
        f'{prefix}/active_latent_dims', active_latent_dims(kl_per_dim, free_bits=free_bits), step
    )
    for index, nats in enumerate(kl_per_dim.tolist()):
        writer.add_scalar(f'{prefix}/kl_per_dim/{index:02d}', nats, step)
