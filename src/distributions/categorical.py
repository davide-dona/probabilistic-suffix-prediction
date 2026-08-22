from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Categorical:
    """A distribution over the latent's modes, `[batch_size, num_modes]`.

    Held as logits rather than probabilities, so the log-probabilities the marginal likelihood
    is assembled from are read off a `log_softmax` instead of a log of a softmax.
    """

    logits: torch.Tensor

    def log_probs(self) -> torch.Tensor:
        """`[batch_size, num_modes]`, the log-probability of each mode."""
        return torch.log_softmax(input=self.logits, dim=-1)

    def probs(self) -> torch.Tensor:
        """`[batch_size, num_modes]`, the probability of each mode."""
        return torch.softmax(input=self.logits, dim=-1)

    def entropy(self) -> torch.Tensor:
        """`[batch_size]`, the entropy of each row in nats.

        Runs from 0, one mode carrying everything, up to `log(num_modes)` for a uniform row, so
        its exponential reads as the number of modes the row spreads over.
        """
        log_probs = self.log_probs()
        return -(log_probs.exp() * log_probs).sum(dim=-1)

    def sample(self, count: int) -> torch.Tensor:
        """Draw `count` modes per row, independently and with replacement.

        What `generate` draws with. Independent draws are what leaves a prefix's suffixes a
        sample of the distribution the model claims, so a mode the prefix makes unlikely turns up
        as rarely as the model says it should rather than once in every `count`.

        Args:
            count: How many modes to draw per row. Unrelated to `num_modes`, since the draws are
                with replacement.
        Returns:
            `[batch_size, count]`, mode indices in the order they were drawn.
        """
        return torch.multinomial(self.probs(), num_samples=count, replacement=True)

    def mode(self) -> torch.Tensor:
        """`[batch_size]`, the most probable mode of each row: the model's single answer."""
        return self.logits.argmax(dim=-1)
