from dataclasses import dataclass

import torch
from torch import nn

from src.configs.schema import Conditioning


@dataclass(frozen=True)
class Modulation:
    """What the latent does to one sublayer: it scales and shifts the input the sublayer reads,
    and scales the branch the sublayer contributes back to the residual stream.

    Each field is `[batch_size, 1, d_model]`, the middle axis broadcasting over positions: z says
    one thing about the whole suffix, so every position of a sequence is modulated alike.
    """

    scale: torch.Tensor
    shift: torch.Tensor
    gate: torch.Tensor


@dataclass(frozen=True)
class LayerModulation:
    """What the latent does to one decoder layer, a `Modulation` per sublayer.

    A mechanism that does not modulate leaves every field None, which `modulate` and `gate` read
    as the identity, so a layer is written once and does not ask which mechanism it is under.
    """

    self_attention: Modulation | None
    cross_attention: Modulation | None
    feedforward: Modulation | None


def modulate(hidden: torch.Tensor, modulation: Modulation | None) -> torch.Tensor:
    """Scale and shift a sublayer's input by what the latent says.

    Args:
        hidden: The normalized input, `[batch_size, seq_len, d_model]`.
        modulation: What to apply, or None to leave `hidden` alone.
    Returns:
        `hidden`, the shape it came in at.
    """
    if modulation is None:
        return hidden
    return hidden * modulation.scale + modulation.shift


def gate(branch: torch.Tensor, modulation: Modulation | None) -> torch.Tensor:
    """Scale what a sublayer contributes back to the residual stream.

    Args:
        branch: The sublayer's output, `[batch_size, seq_len, d_model]`.
        modulation: What to apply, or None to leave `branch` alone.
    Returns:
        `branch`, the shape it came in at.
    """
    if modulation is None:
        return branch
    return branch * modulation.gate


class LatentConditioning(nn.Module):
    """How the latent reaches the decoder.

    One subclass per mechanism, and the decoder holds exactly one of them: it calls `prefix` where
    the cross-attention source is assembled and `layers` once per pass, whichever mechanism it was
    built with. Everything a mechanism does lives in its subclass, so the decoder itself never
    asks which one it has.

    Attributes:
        affine_norms: Whether the decoder's layers give their pre-norms a learned affine of their
            own. False where the mechanism supplies the scale and shift itself.
    """

    affine_norms: bool

    def prefix(
        self, z: torch.Tensor, prefix_encoded: torch.Tensor, prefix_pad_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The cross-attention source the decoder reads, given the latent.

        Args:
            z: The sampled latent, `[batch_size, latent_dim]`.
            prefix_encoded: The encoded prefix events, `[batch_size, prefix_seq_len, d_model]`.
            prefix_pad_mask: True where a prefix position holds padding.
        Returns:
            The source to cross-attend over and its padding mask.
        """
        raise NotImplementedError

    def layers(self, z: torch.Tensor) -> tuple[LayerModulation, ...]:
        """What the latent does to each decoder layer.

        Read once per pass rather than once per decode step: z is fixed for the whole of a
        generation, so a step costs nothing beyond applying what this returned.

        Args:
            z: The sampled latent, `[batch_size, latent_dim]`.
        Returns:
            One `LayerModulation` per decoder layer, in stack order.
        """
        raise NotImplementedError


class TokenConditioning(LatentConditioning):
    """z as an extra cross-attention token, prepended to the encoded prefix.

    Every position reads it with a content-dependent weight rather than a fixed additive bias, but
    that weight is a softmax against the prefix's own positions and nothing holds it away from
    zero.
    """

    affine_norms = True

    def __init__(self, *, latent_dim: int, d_model: int, num_layers: int):
        super().__init__()
        # Brings the latent up to the width of the cross-attention token it becomes.
        self.latent_projection = nn.Linear(in_features=latent_dim, out_features=d_model)
        # No layer modulation, so one value serves every layer and every batch.
        self._layers = (LayerModulation(None, None, None),) * num_layers

    def prefix(
        self, z: torch.Tensor, prefix_encoded: torch.Tensor, prefix_pad_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepend the projected latent to the prefix as an extra cross-attention token.

        Args:
            z: The sampled latent, `[batch_size, latent_dim]`.
            prefix_encoded: The encoded prefix events, `[batch_size, prefix_seq_len, d_model]`.
            prefix_pad_mask: True where a prefix position holds padding.
        Returns:
            `prefix_encoded` and `prefix_pad_mask`, each one position longer, with the latent
            token first.
        """
        latent_token = self.latent_projection(z).unsqueeze(dim=1)  # [batch_size, 1, d_model]
        prefix_encoded = torch.cat(
            tensors=(latent_token, prefix_encoded), dim=1
        )  # [batch_size, 1 + prefix_seq_len, d_model]
        # The latent token is never padding, so add a column of False to match the new width.
        latent_column = prefix_pad_mask.new_zeros(size=(prefix_pad_mask.size(dim=0), 1))
        prefix_pad_mask = torch.cat(
            tensors=(latent_column, prefix_pad_mask), dim=1
        )  # [batch_size, 1 + prefix_seq_len]
        return prefix_encoded, prefix_pad_mask

    def layers(self, z: torch.Tensor) -> tuple[LayerModulation, ...]:
        """One unmodulated `LayerModulation` per layer: this mechanism works on the prefix alone."""
        return self._layers


class AdaLNConditioning(LatentConditioning):
    """z as an adaptive layer norm: it scales and shifts every sublayer's input and gates every
    sublayer's output, in every layer.

    The prefix is left alone. Unlike a cross-attention token, this path cannot be attenuated by an
    attention weight: a position's representation passes through the modulation whatever the
    decoder has learned to attend to, which is what keeps z carrying information rather than
    collapsing onto the prior.

    Each layer reads nine `[batch_size, d_model]` vectors off its own head: a scale, a shift and a
    gate for each of the three sublayers. The head is zero-initialized and the scale and gate are
    read as `1 + w(z)`, so a layer starts as exactly the unconditioned pre-norm block it would
    otherwise be, and the modulation is learned from there rather than disturbing the stack at
    step zero.
    """

    affine_norms = False

    # Per sublayer: a scale, a shift and a gate; per layer: three sublayers.
    _PARAMETERS_PER_SUBLAYER = 3
    _SUBLAYERS = 3

    def __init__(self, *, latent_dim: int, d_model: int, num_layers: int):
        super().__init__()
        self.d_model = d_model
        # A nonlinear map of z, shared by every layer: read straight off `latent_dim` instead, the
        # modulation each layer applies could only ever be a linear function of the latent.
        self.trunk = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=d_model), nn.SiLU()
        )
        width = self._PARAMETERS_PER_SUBLAYER * self._SUBLAYERS * d_model
        self.heads = nn.ModuleList(
            nn.Linear(in_features=d_model, out_features=width) for _ in range(num_layers)
        )
        # Zeroed weight and bias alike, so every head emits zeros and every layer starts at
        # scale 1, shift 0, gate 1. The trunk's output is not zero, so the heads still take
        # gradient from the first step.
        for head in self.heads:
            nn.init.zeros_(tensor=head.weight)
            nn.init.zeros_(tensor=head.bias)

    def prefix(
        self, z: torch.Tensor, prefix_encoded: torch.Tensor, prefix_pad_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The encoded prefix unchanged: this mechanism carries the latent inside the layers.

        Args:
            z: The sampled latent, `[batch_size, latent_dim]`.
            prefix_encoded: The encoded prefix events, `[batch_size, prefix_seq_len, d_model]`.
            prefix_pad_mask: True where a prefix position holds padding.
        Returns:
            Its `prefix_encoded` and `prefix_pad_mask` arguments.
        """
        return prefix_encoded, prefix_pad_mask

    def layers(self, z: torch.Tensor) -> tuple[LayerModulation, ...]:
        """Read every layer's modulation off the latent.

        Args:
            z: The sampled latent, `[batch_size, latent_dim]`.
        Returns:
            One `LayerModulation` per decoder layer, in stack order.
        """
        conditioning = self.trunk(z)  # [batch_size, d_model]
        return tuple(self._modulations(head(conditioning)) for head in self.heads)

    def _modulations(self, parameters: torch.Tensor) -> LayerModulation:
        """Split one head's output into a `Modulation` per sublayer.

        Args:
            parameters: One layer's head output, `[batch_size, 9 * d_model]`.
        Returns:
            The layer's modulation, in sublayer order: self-attention, cross-attention,
            feed-forward.
        """
        # [batch_size, 9, d_model] -> a [batch_size, 1, d_model] slice per parameter, the middle
        # axis left in place so each broadcasts over the positions it modulates.
        split = parameters.unflatten(
            dim=-1, sizes=(self._PARAMETERS_PER_SUBLAYER * self._SUBLAYERS, self.d_model)
        ).unsqueeze(dim=2)  # [batch_size, 9, 1, d_model]
        return LayerModulation(
            *(
                Modulation(
                    scale=1.0 + split[:, index],
                    shift=split[:, index + 1],
                    gate=1.0 + split[:, index + 2],
                )
                for index in range(0, self._PARAMETERS_PER_SUBLAYER * self._SUBLAYERS, 3)
            )
        )


def build_conditioning(
    *, conditioning: Conditioning, latent_dim: int, d_model: int, num_layers: int
) -> LatentConditioning:
    """The mechanism a config names, built for a decoder of `num_layers` layers.

    Args:
        conditioning: What `DecoderConfig.conditioning` declared.
        latent_dim: The width of the latent the mechanism reads.
        d_model: The decoder's width.
        num_layers: How many layers the mechanism has to serve.
    Returns:
        The mechanism, ready for the decoder to hold.
    """
    mechanisms: dict[Conditioning, type[LatentConditioning]] = {
        'token': TokenConditioning,
        'adaln': AdaLNConditioning,
    }
    return mechanisms[conditioning](latent_dim=latent_dim, d_model=d_model, num_layers=num_layers)
