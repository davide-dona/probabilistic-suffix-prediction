from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from src.datasets.dataset import SplitTrace
from src.model import TransformerCVAEOutput
from src.scalar_metrics import ScalarMetrics

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


@dataclass(frozen=True)
class Loss(ScalarMetrics):
    """The loss of one pass, and the terms it is made of.

    `loss` is the exact negative log-likelihood the pass is trained on; the two terms beside it
    are what the modes reconstructed, each weighed by the responsibility that mode took, so they
    stay readable in the nats `loss` is in without adding up to it.
    """

    loss: float = 0.0
    activity_loss: float = 0.0
    remaining_time_loss: float = 0.0


@dataclass(frozen=True, slots=True)
class LatentUsage:
    """How the pass spread itself over the modes.

    A categorical latent fails by leaving modes unused rather than by collapsing onto its prior,
    so what has to be watched is how many of them carry anything and how much the suffix narrows
    the choice. Both are read off the responsibilities `q(k | prefix, suffix)`, which is the
    exact posterior here rather than an amortized stand-in for it.

    Every field is summed over the traces of the pass, like `Loss`, so batches are accumulated
    with `+` and divided once by the traces they covered. It is logged rather than reported, so,
    like `Loss`, it declares no unit; unlike `Loss` it holds a vector and reads entropies off
    aggregates, neither of which `ScalarMetrics` is float-only for.
    """

    # A 0-d zero is an empty total: it broadcasts against the first vector added to it, so an
    # accumulator needs to know nothing of how many modes there are, and torch adds a 0-d CPU
    # tensor to a tensor on any device.
    responsibilities: torch.Tensor = torch.zeros(())
    posterior_entropy: torch.Tensor = torch.zeros(())
    prior_entropy: torch.Tensor = torch.zeros(())

    def __add__(self, other: 'LatentUsage') -> 'LatentUsage':
        return LatentUsage(
            responsibilities=self.responsibilities + other.responsibilities,
            posterior_entropy=self.posterior_entropy + other.posterior_entropy,
            prior_entropy=self.prior_entropy + other.prior_entropy,
        )

    def __truediv__(self, divisor: float) -> 'LatentUsage':
        return LatentUsage(
            responsibilities=self.responsibilities / divisor,
            posterior_entropy=self.posterior_entropy / divisor,
            prior_entropy=self.prior_entropy / divisor,
        )

    def log(self, writer: 'SummaryWriter', step: int, *, prefix: str) -> None:
        """Write how many modes the pass used, and how much the suffix said about which.

        Read of a pass already divided by the traces it covered, since `responsibilities` is the
        mean posterior over the split and the two entropies are per-trace means.

        Args:
            writer: The TensorBoard writer to log to.
            step: The step these metrics belong to.
            prefix: Namespace to log under, e.g. `val`, matching the one `Loss` is logged under.
        """
        # How wide the split's own choice of mode is, whoever made it. Its exponential is the
        # number of modes that are actually being written under, from 1 up to `num_modes`.
        marginal_entropy = _entropy(self.responsibilities)

        writer.add_scalar(f'{prefix}/mode_perplexity', marginal_entropy.exp().item(), step)
        # What the suffix tells us about the mode: the split's spread over modes, less what is
        # left of it once the suffix is known. Zero means the suffix picks no mode in particular.
        writer.add_scalar(
            f'{prefix}/mutual_information',
            (marginal_entropy - self.posterior_entropy).item(),
            step,
        )
        writer.add_scalar(f'{prefix}/posterior_entropy', self.posterior_entropy.item(), step)
        writer.add_scalar(f'{prefix}/prior_entropy', self.prior_entropy.item(), step)


def compute_loss(
    output: TransformerCVAEOutput,
    batch: SplitTrace,
    *,
    pad_activity_index: int,
) -> tuple[torch.Tensor, Loss, LatentUsage]:
    """Score a forward pass's predictions against the batch it was run on.

    The pass wrote the suffix under every mode, so what it is scored on is the exact marginal
    likelihood `sum_k p(k | prefix) * p(suffix | prefix, k)`: no bound, no divergence term, and
    so nothing that charges the model for telling the modes apart. The posterior over modes falls
    out of the same numbers as a softmax and is read for the diagnostics alone.

    Args:
        output: The model's prediction for `batch`, from `model(batch)`.
        batch: A batch from `TraceDataset`, already on the right device.
        pad_activity_index: `model.pad_activity_index`, ignored in the activity cross-entropy.
    Returns:
        The per-trace loss to backpropagate, the metrics to log, and how the pass spread itself
        over the modes, the last two summed over the batch.
    """
    batch_size, num_modes = output.prior.logits.shape

    # Cross-entropy reads one row per position, so the modes are folded back into the batch axis
    # for it and split out again after. The targets repeat mode-fastest, the layout the model
    # laid its rows out in.
    logits = output.decoder.activity_logits.flatten(start_dim=0, end_dim=1)
    targets = batch.suffix.activities.repeat_interleave(repeats=num_modes, dim=0)
    activity_nll = (
        F.cross_entropy(
            logits.transpose(1, 2),
            targets,
            ignore_index=pad_activity_index,
            reduction='none',
        )
        .sum(dim=1)
        .view(batch_size, num_modes)
    )  # [batch_size, num_modes]

    remaining_time_nll = output.decoder.remaining_time_distr.nll(
        batch.remaining_time.unsqueeze(dim=1)
    )  # [batch_size, num_modes]

    # log p(k | prefix) + log p(suffix | prefix, k), the joint each mode contributes.
    log_joint = output.prior.log_probs() - (
        activity_nll + remaining_time_nll
    )  # [batch_size, num_modes]
    total_loss = -torch.logsumexp(log_joint, dim=1).sum()

    # q(k | prefix, suffix), exactly: the joint renormalized over the modes. Read by the
    # diagnostics and by the two reconstruction terms, so it is built off the graph.
    with torch.no_grad():
        responsibilities = log_joint.softmax(dim=1)  # [batch_size, num_modes]
        metrics = Loss(
            loss=total_loss.item(),
            activity_loss=(responsibilities * activity_nll).sum().item(),
            remaining_time_loss=(responsibilities * remaining_time_nll).sum().item(),
        )
        latent_usage = LatentUsage(
            responsibilities=responsibilities.sum(dim=0),  # [num_modes]
            posterior_entropy=_entropy(responsibilities).sum(),
            prior_entropy=output.prior.entropy().sum(),
        )
    return total_loss / batch_size, metrics, latent_usage


def _entropy(probabilities: torch.Tensor) -> torch.Tensor:
    """The entropy of each row of `probabilities`, in nats.

    Args:
        probabilities: `[batch_size, num_modes]`, each row summing to one.
    Returns:
        `[batch_size]`. A mode of probability 0 contributes nothing, which the log is floored to
        keep it from being a NaN instead.
    """
    floored = probabilities.clamp(min=torch.finfo(probabilities.dtype).tiny)
    return -(probabilities * floored.log()).sum(dim=-1)
