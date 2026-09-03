from dataclasses import dataclass

import torch

# The head emits the log of its scale rather than the scale itself, so the scale stays positive by
# construction. The range is clamped because `exp` of an unbounded head turns into an inf, and
# from there into a NaN loss, long before training would recover.
LOGSCALE_MIN, LOGSCALE_MAX = -10.0, 10.0


@dataclass(frozen=True)
class Laplace:
    """A Laplace distribution, `[...]` per field, parameterized by its median and its log-scale.

    What both time heads are written in. The family is chosen for what its mode is: the minimizer
    of `|mean - target|` is the conditional *median*, and the median is what the report's
    `*_ae_*_days` columns score. A squared error would fit the conditional mean instead, over
    durations that are standardized raw minutes with nothing but a 99.9-percentile filter guarding
    their tail, and the tail would set the fit.

    A head that emits its own scale can buy its way out of a hard position by widening, which is
    the price of letting it be drawn from: a model whose draws come from its heads has to say how
    wide they are. A head that is read at its mode instead pins `logscale` to 0, which leaves the
    negative log-likelihood the plain absolute error, and leaves `sample` nothing to add beyond
    unit-scale noise no caller asks for.
    """

    mean: torch.Tensor
    logscale: torch.Tensor

    @classmethod
    def create(cls, mean: torch.Tensor, logscale: torch.Tensor) -> 'Laplace':
        """Build a Laplace from an already-split median and log-scale.

        Args:
            mean: The distribution's median, which is also its mode and its mean.
            logscale: The raw log-scale, clamped to `[LOGSCALE_MIN, LOGSCALE_MAX]` before it is
                stored, so a later `exp()` cannot overflow.
        Returns:
            The Laplace with its log-scale clamped.
        """
        return cls(mean=mean, logscale=logscale.clamp(min=LOGSCALE_MIN, max=LOGSCALE_MAX))

    @classmethod
    def point(cls, mean: torch.Tensor) -> 'Laplace':
        """A head read at its mode alone, its scale pinned to 1.

        The negative log-likelihood of this is exactly `|mean - target|`, so a decoder whose
        variability lives elsewhere is scored by the same call as one whose variability lives in
        its heads, without the loss having to ask which it is holding.

        Args:
            mean: The head's output, `[...]`.
        Returns:
            The unit-scale Laplace around it.
        """
        return cls(mean=mean, logscale=torch.zeros_like(input=mean))

    def negative_log_likelihood(self, target: torch.Tensor) -> torch.Tensor:
        """How badly this distribution accounts for `target`, without the `log 2` both halves of a
        comparison would carry.

        Args:
            target: What the distribution is scored against, the shape of `mean`.
        Returns:
            The per-element negative log-likelihood, `[...]`.
        """
        # exp(-logscale) rather than a division by the scale: one exp and a multiply, and no
        # reciprocal of a number that a wide head can drive towards 0.
        return self.logscale + (self.mean - target).abs() * torch.exp(input=-self.logscale)

    def sample(self) -> torch.Tensor:
        """Draw one sample by inverse transform, so the gradient reaches `mean` and `logscale`.

        Returns:
            One draw, `[...]`, shaped like `mean`.
        """
        # The Laplace quantile function, `median - scale * sign(u) * log1p(-2 * |u|)` for
        # `u ~ U(-0.5, 0.5)`. As with the Gaussian's reparametrization the randomness sits in a
        # term with no parameters, which is what leaves a gradient path through both fields.
        uniform = torch.rand_like(input=self.mean) - 0.5  # [...], in [-0.5, 0.5)
        # `torch.rand` draws from [0, 1), so `uniform` reaches exactly -0.5 and `log1p(-1)` is an
        # -inf. Half an ulp inside the interval is what keeps the draw finite.
        half = torch.tensor(data=0.5, dtype=uniform.dtype, device=uniform.device)
        limit = torch.nextafter(input=half, other=torch.zeros_like(input=half))
        uniform = uniform.clamp(min=-limit, max=limit)

        scale = torch.exp(input=self.logscale)  # [...]
        return self.mean - scale * uniform.sign() * torch.log1p(input=-2.0 * uniform.abs())
