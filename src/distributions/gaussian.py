import math
from dataclasses import dataclass

import torch

# The latent heads emit a log-variance rather than a standard deviation, so the scale stays
# positive by construction. The range is clamped because `exp` of an unbounded head turns
# into an inf, and from there into a NaN loss, long before training would recover.
LOGVAR_MIN, LOGVAR_MAX = -10.0, 10.0


@dataclass(frozen=True)
class Gaussian:
    """A diagonal Gaussian, `[batch_size, dim]` per field."""

    mean: torch.Tensor
    logvar: torch.Tensor

    @classmethod
    def create(cls, mean: torch.Tensor, logvar: torch.Tensor) -> 'Gaussian':
        """Build a Gaussian from an already-split mean and log-variance.

        Args:
            mean: The distribution's mean.
            logvar: The raw log-variance, clamped to `[LOGVAR_MIN, LOGVAR_MAX]` before it is
                stored, so a later `exp()` cannot overflow.
        Returns:
            The Gaussian with its log-variance clamped.
        """
        return cls(mean=mean, logvar=logvar.clamp(min=LOGVAR_MIN, max=LOGVAR_MAX))

    @classmethod
    def from_head(cls, parameters: torch.Tensor) -> 'Gaussian':
        """Build a Gaussian from a head that emits both halves side by side.

        The convention every latent head follows: the first half of its output is the mean, the
        second the raw log-variance. Written down once here, so a head is read the same way
        wherever one is added.

        Args:
            parameters: A head's output, `[batch_size, 2 * dim]`.
        Returns:
            The Gaussian those halves describe, its log-variance clamped as `create` clamps it.
        """
        mean, logvar = parameters.chunk(chunks=2, dim=-1)  # each [batch_size, dim]
        return cls.create(mean=mean, logvar=logvar)

    def nll(self, target: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood of a target under this Gaussian, one value per element.

        Left unreduced because the caller decides what a row is: the marginal likelihood weighs
        each mode's negative log-likelihood before anything is summed.

        Args:
            target: The observed value, broadcast against `mean`.
        Returns:
            The negative log-likelihoods, the shape `mean` came in at.
        """
        return 0.5 * (
            self.logvar + (target - self.mean) ** 2 / self.logvar.exp() + math.log(2.0 * math.pi)
        )
