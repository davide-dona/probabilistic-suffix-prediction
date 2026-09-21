from dataclasses import dataclass

import torch

# Log-scale keeps the scale positive. Clamping keeps exp finite.
LOGSCALE_MIN, LOGSCALE_MAX = -10.0, 10.0


@dataclass(frozen=True)
class Laplace:
    """A Laplace distribution, [...] per field, parameterized by median and log-scale.

    It handles occasional long durations better than a Gaussian.
    """

    mean: torch.Tensor
    logscale: torch.Tensor

    @classmethod
    def create(cls, mean: torch.Tensor, logscale: torch.Tensor) -> 'Laplace':
        """Build a Laplace from a median and log-scale.

        Args:
            mean: The distribution's median.
            logscale: The raw log-scale.
        Returns:
            The Laplace with a clamped log-scale.
        """
        return cls(mean=mean, logscale=logscale.clamp(min=LOGSCALE_MIN, max=LOGSCALE_MAX))

    @classmethod
    def point(cls, mean: torch.Tensor) -> 'Laplace':
        """Build a unit-scale Laplace centered on the median.

        Args:
            mean: The distribution's median, [...].
        Returns:
            The unit-scale Laplace around it.
        """
        return cls(mean=mean, logscale=torch.zeros_like(input=mean))

    def beta_nll(self, target: torch.Tensor) -> torch.Tensor:
        """Return beta-NLL while keeping the median gradient equal to absolute error.

        Args:
            target: What the distribution is scored against, the shape of mean.
        Returns:
            The per-element objective, [...].
        """
        likelihood = self.logscale + (self.mean - target).abs() * torch.exp(input=-self.logscale)
        # Detaching the scale keeps the median gradient equal to absolute error.
        return torch.exp(input=self.logscale).detach() * likelihood

    def scale_penalty(self) -> torch.Tensor:
        """Return the scale contribution to beta-NLL.

        Returns:
            The per-element term, [...].
        """
        return torch.exp(input=self.logscale) * self.logscale

    def sample(self) -> torch.Tensor:
        """Draw one differentiable sample.

        Returns:
            One draw, [...], shaped like mean.
        """
        # Inverse-transform sampling keeps gradients through both parameters.
        uniform = torch.rand_like(input=self.mean) - 0.5  # [...], in [-0.5, 0.5)
        # Clamp the endpoint so log1p stays finite.
        half = torch.tensor(data=0.5, dtype=uniform.dtype, device=uniform.device)
        limit = torch.nextafter(input=half, other=torch.zeros_like(input=half))
        uniform = uniform.clamp(min=-limit, max=limit)

        scale = torch.exp(input=self.logscale)  # [...]
        return self.mean - scale * uniform.sign() * torch.log1p(input=-2.0 * uniform.abs())
