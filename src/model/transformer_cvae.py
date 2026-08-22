from dataclasses import dataclass

import torch
from torch import nn

from src.configs.schema import ModelConfig
from src.datasets.codec import DatasetCodec
from src.datasets.dataset import SplitTrace
from src.distributions.categorical import Categorical
from src.model.checkpoint import MODEL_KEYS, require_keys
from src.model.components.decoder import Decoder, DecoderOutput, GeneratedSuffix
from src.model.components.embeddings import EventEmbeddings
from src.model.components.latent import PriorNetwork
from src.model.components.trace_encoder import TraceEncoder


@dataclass(frozen=True)
class TransformerCVAEOutput:
    """One pass over a batch: what the decoder wrote under every mode, and how likely the
    prefix alone makes each of them.

    The decoder's fields carry `[batch_size, num_modes, ...]`, one row per trace per mode, which
    is what lets the loss weigh a mode's reconstruction by the prior probability beside it.
    """

    decoder: DecoderOutput
    prior: Categorical


class TransformerCVAE(nn.Module):
    """A conditional VAE that predicts a trace's suffix from its prefix.

    The latent is a choice among `num_modes` ways the trace can carry on, which is what a suffix
    is: a path through the process, not a point on a continuum. The prefix is the condition: the
    decoder cross-attends over its encoded events, and its CLS summary says how likely the prefix
    makes each mode. Because the decoder reads the prefix directly and the prior is conditioned
    on it too, a mode is left carrying only what the prefix does not determine.

    Finitely many modes are few enough to enumerate, so training writes the suffix under every
    one of them and the loss marginalizes; the posterior over modes is then a closed-form
    function of what came back and never has to be amortized, which is why there is no suffix
    encoder here.

    Flow, with `prefix` and `suffix` both padded to `max_seq_len`:
        prefix                       -> prefix events (for the decoder) + summary (for the prior)
        prefix summary               -> p(k | prefix)
        every k, prefix events, suffix -> activity predictions, and the case's remaining time
    """

    def __init__(self, config: ModelConfig, codec: DatasetCodec):
        super().__init__()
        # Shared between both encoders and the decoder: a single embedding space for events,
        # with fewer parameters.
        self.embeddings = EventEmbeddings(
            config=config.embeddings, codec=codec, d_model=config.d_model
        )
        self.prefix_encoder = TraceEncoder(
            config=config.prefix_encoder, embeddings=self.embeddings, d_model=config.d_model
        )
        self.prior = PriorNetwork(
            config=config.prior, latent_config=config.latent, prefix_dim=config.d_model
        )
        self.decoder = Decoder(
            config=config.decoder,
            latent_config=config.latent,
            embeddings=self.embeddings,
            d_model=config.d_model,
            num_activities=codec.activity.num_rows,
            sos_activity_index=codec.activity.sos_index,
            pad_activity_index=codec.activity.pad_index,
            pad_resource_index=codec.resource.pad_index,
            eot_activity_index=codec.activity.eot_index,
        )
        self.num_modes = config.latent.num_modes
        self.pad_activity_index = codec.activity.pad_index
        self.eot_activity_index = codec.activity.eot_index

    @classmethod
    def from_checkpoint(
        cls, checkpoint: dict, codec: DatasetCodec, *, device: str = 'cpu'
    ) -> 'TransformerCVAE':
        """Rebuild the model a checkpoint holds, with its weights loaded.

        Args:
            checkpoint: A checkpoint read by `load_checkpoint`.
            codec: The dataset the model is to be used on, supplying the vocabulary
                sizes and sequence length it was built against.
            device: Where to place the model.
        Returns:
            The model, in evaluation mode.
        Raises:
            ValueError: If the checkpoint does not carry a config and weights.
        """
        require_keys(checkpoint, MODEL_KEYS, purpose='rebuilt', remedy='Train the model again.')
        config = ModelConfig.model_validate(checkpoint['model_config'])

        model = cls(config=config, codec=codec).to(device=device)
        model.load_state_dict(state_dict=checkpoint['model_state_dict'])
        model.eval()
        return model

    def forward(self, item: SplitTrace) -> TransformerCVAEOutput:
        """Write the batch's suffix under every mode at once.

        Args:
            item: A batch from `TraceDataset`, read for both its prefix and its suffix.
        Returns:
            What each mode reconstructed, `[batch_size, num_modes, ...]`, beside the prior the
            loss weighs those reconstructions by.
        """
        prefix_pad_mask = item.prefix.pad_mask()  # [batch_size, seq_len]
        prefix = self.prefix_encoder(events=item.prefix, pad_mask=prefix_pad_mask)

        # The decoder reads the prefix events only; the summary belongs to the latent path.
        prior = self.prior(prefix.summary)  # p(k | prefix), [batch_size, num_modes]

        # Every trace is written under every mode, which is what the loss marginalizes over.
        modes = torch.arange(end=self.num_modes, device=prefix.events.device).expand(
            item.prefix.length.size(dim=0), self.num_modes
        )
        return TransformerCVAEOutput(
            decoder=self.decoder(
                suffix_activities=item.suffix.activities,
                modes=modes,
                prefix_encoded=prefix.events,
                prefix_pad_mask=prefix_pad_mask,
            ),
            prior=prior,
        )

    @torch.no_grad()
    def generate(
        self, item: SplitTrace, *, num_samples: int, sample_latent: bool = True
    ) -> GeneratedSuffix:
        """Generate `num_samples` suffixes for every prefix in `item`.

        The suffix is unknown here, so only the prefix is encoded and every mode is drawn from
        `p(k | prefix)`. The samples of one prefix differ only in that mode, since the decoder
        reads its heads greedily.

        Args:
            item: A batch from `TraceDataset`, read for its prefix only.
            num_samples: How many suffixes to draw per prefix.
            sample_latent: Whether to draw the modes from `p(k | prefix)`. False takes its most
                probable mode instead, making the generation the model's single point prediction.
        Returns:
            The generated suffixes, `[batch_size, num_samples, steps]`, with row `(i, j)` the
            j-th sample for the i-th prefix of the batch.
        """
        prefix_pad_mask = item.prefix.pad_mask()  # [batch_size, seq_len]
        prefix = self.prefix_encoder(events=item.prefix, pad_mask=prefix_pad_mask)

        # Computed once per prefix: every sample of one prefix is drawn from the same
        # p(k | prefix), so running the prior once and drawing from it skips num_samples - 1
        # redundant forward passes over identical rows.
        prior = self.prior(prefix.summary)
        modes = (
            prior.sample(num_samples)
            if sample_latent
            else prior.mode().unsqueeze(dim=1).expand(-1, num_samples)
        )  # [batch_size, num_samples]

        # A suffix holds at most `max_seq_len` events, the padded width the batch comes in at.
        return self.decoder.generate(
            modes=modes,
            prefix_encoded=prefix.events,
            prefix_pad_mask=prefix_pad_mask,
            max_steps=item.prefix.activities.size(dim=1),
        )
