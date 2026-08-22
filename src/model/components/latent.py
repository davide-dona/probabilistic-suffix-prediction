import torch
from torch import nn

from src.configs.schema import LatentConfig, PriorConfig
from src.distributions.categorical import Categorical


class PriorNetwork(nn.Module):
    """`p(k | prefix)`: how likely each of the latent's modes is, given the prefix.

    A mode is one way the trace can carry on, and this is the only distribution the model
    carries: with finitely many modes the posterior `q(k | prefix, suffix)` is a closed-form
    function of this and of what each mode reconstructs, so nothing has to be amortized for it.
    """

    def __init__(self, config: PriorConfig, latent_config: LatentConfig, *, prefix_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        # The input is the prefix summary; each hidden layer then narrows or widens from there.
        width = prefix_dim
        for hidden_dim in config.hidden_dims:
            layers += [
                nn.Linear(in_features=width, out_features=hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=config.dropout),
            ]
            width = hidden_dim
        # The output layer emits one logit per mode.
        layers.append(nn.Linear(in_features=width, out_features=latent_config.num_modes))

        # With `hidden_dims` empty this collapses to a single linear layer.
        self.net = nn.Sequential(*layers)

    def forward(self, prefix_summary: torch.Tensor) -> Categorical:
        """
        Args:
            prefix_summary: The prefix encoder's summary, `[batch_size, d_model]`.
        Returns:
            `p(k | prefix)`, `[batch_size, num_modes]`.
        """
        return Categorical(logits=self.net(prefix_summary))
