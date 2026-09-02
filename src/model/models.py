from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from pydantic import TypeAdapter
from torch import nn

from src.configs.schema import CVAEConfig, ModelConfig, TransformerConfig
from src.datasets.codec import DatasetCodec
from src.datasets.dataset import SplitTrace
from src.distributions.gaussian import Gaussian
from src.model.checkpoint import MODEL_KEYS, require_keys
from src.model.components.decoder import Decoder, DecoderOutput, GeneratedSuffix
from src.model.components.embeddings import EventEmbeddings
from src.model.components.latent import PosteriorNetwork, PriorNetwork
from src.model.components.trace_encoder import TraceEncoder

# `ModelConfig` is a tagged union rather than a class, so a checkpoint's stored config is
# validated through an adapter rather than by calling `model_validate` on it.
_MODEL_CONFIG = TypeAdapter(ModelConfig)


@dataclass(frozen=True)
class Latents:
    """The two distributions the KL term compares, produced only by a model that has a latent."""

    prior: Gaussian  # p(z | prefix)
    posterior: Gaussian  # q(z | prefix, suffix)


@dataclass(frozen=True)
class ModelOutput:
    """What one training pass produced, whichever architecture ran it.

    `latents` is the whole of the difference the loss sees: with them the pass is scored by the
    ELBO, without them by its reconstruction alone.
    """

    decoder: DecoderOutput
    latents: Latents | None  # None for a model with no latent


class SuffixModel(nn.Module, ABC):
    """What the training loop, the validation pass and the generation pipeline ask of a model.

    Deliberately narrow: one pass, one generation, and the padding index the loss ignores.
    Everything that separates the architectures - where the variability lives, how the heads are
    read, what the loss charges - is answered behind this rather than branched on in front of it.
    """

    def __init__(self, codec: DatasetCodec):
        super().__init__()
        # Read off the codec rather than passed down: which index means padding is a property of
        # the dataset, and every model built against one agrees about it.
        self.pad_activity_index = codec.activity.pad_index
        self.eot_activity_index = codec.activity.eot_index

    @abstractmethod
    def forward(self, item: SplitTrace) -> ModelOutput:
        """Score one batch teacher-forced, for the loss to charge."""

    @abstractmethod
    def generate(
        self, item: SplitTrace, *, num_samples: int, sample: bool = True
    ) -> GeneratedSuffix:
        """Write `num_samples` suffixes for every prefix of a batch.

        Args:
            item: A batch from `TraceDataset`, read for its prefix only.
            num_samples: How many suffixes to draw per prefix.
            sample: Whether this is a draw rather than the model's single point prediction. What
                is drawn from is the architecture's business.
        Returns:
            The suffixes, `[batch_size, num_samples, ...]`.
        """

    def _per_sample(self, generated: GeneratedSuffix, *, batch_size: int) -> GeneratedSuffix:
        """Split a decoder's flat rows back into the prefix each belongs to.

        Args:
            generated: What `Decoder.generate` wrote for `batch_size * num_samples` rows, the
                samples of one prefix adjacent, as `repeat_interleave` laid them out.
            batch_size: How many prefixes those rows came from.
        Returns:
            The same suffixes as `[batch_size, num_samples, ...]`.
        """
        return GeneratedSuffix(
            activities=generated.activities.view(batch_size, -1, generated.activities.size(dim=1)),
            lengths=generated.lengths.view(batch_size, -1),
            times_to_next=generated.times_to_next.view(
                batch_size, -1, generated.times_to_next.size(dim=1)
            ),
            remaining_time=generated.remaining_time.view(batch_size, -1),
        )


class TransformerCVAE(SuffixModel):
    """A conditional VAE that predicts a trace's suffix from its prefix.

    The prefix is the condition: the decoder cross-attends over its encoded events, and its
    CLS summary feeds the latent networks. Because the decoder reads the prefix directly and
    the prior is conditioned on it too, z is left encoding only what the prefix does not
    determine.

    One encoder reads both sequences, so the two summaries the posterior is handed are two
    reads of one representation rather than two coordinate systems the KL has to reconcile.

    Flow, with `prefix` and `suffix` both padded to `max_seq_len`:
        prefix              -> prefix events (for the decoder) + summary (for the latents)
        prefix summary      -> p(z | prefix)               (scored by the KL term only)
        + suffix summary    -> q(z | prefix, suffix)
        z ~ q(z | prefix, suffix)
        z, prefix events, suffix -> an activity, the wait until it and a remaining time,
                                   at every suffix position
    """

    def __init__(self, config: CVAEConfig, codec: DatasetCodec):
        super().__init__(codec=codec)
        # Shared between the encoder and the decoder: a single embedding space for events,
        # with fewer parameters.
        self.embeddings = EventEmbeddings(
            config=config.embeddings, codec=codec, d_model=config.d_model
        )
        # One stack for both sequences: a prefix and a suffix are the same kind of thing, read
        # the same way, so a summary of either lands in one coordinate system. Only the prefix
        # is encoded at inference, the suffix being what the model is asked to write.
        self.encoder = TraceEncoder(
            config=config.encoder, embeddings=self.embeddings, d_model=config.d_model
        )
        self.prior = PriorNetwork(
            config=config.prior, latent_config=config.latent, prefix_dim=config.d_model
        )
        self.posterior = PosteriorNetwork(latent_config=config.latent, summary_dim=config.d_model)
        # Greedy heads: everything the prefix does not determine is z's to carry, so a suffix is
        # one draw from the latent rather than a sequence of independent per-step draws.
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
            stochastic=False,
        )

    def forward(self, item: SplitTrace) -> ModelOutput:
        """
        Args:
            item: A batch from `TraceDataset`, read for both its prefix and its suffix.
        Returns:
            The decoder's predictions and the latent distributions the loss compares.
        """
        prefix_pad_mask = item.prefix.pad_mask()  # [batch_size, seq_len]
        prefix = self.encoder(events=item.prefix, pad_mask=prefix_pad_mask)

        suffix_summary = self.encoder(
            events=item.suffix, pad_mask=item.suffix.pad_mask()
        ).summary  # [batch_size, d_model]

        prior = self.prior(prefix.summary)  # p(z | prefix)
        posterior = self.posterior(prefix_summary=prefix.summary, suffix_summary=suffix_summary)
        z = posterior.sample()  # [batch_size, latent_dim]

        # The decoder reads the prefix events only; the summary belongs to the latent path.
        decoder_output = self.decoder(
            suffix_activities=item.suffix.activities,
            z=z,
            prefix_encoded=prefix.events,
            prefix_pad_mask=prefix_pad_mask,
        )
        return ModelOutput(
            decoder=decoder_output, latents=Latents(prior=prior, posterior=posterior)
        )

    @torch.no_grad()
    def generate(
        self, item: SplitTrace, *, num_samples: int, sample: bool = True
    ) -> GeneratedSuffix:
        """Generate `num_samples` suffixes for every prefix in `item`.

        The suffix is unknown here, so only the prefix is encoded and every latent comes from
        `p(z | prefix)`. The samples of one prefix differ only in that latent, since the
        decoder reads its heads greedily.

        Args:
            item: A batch from `TraceDataset`, read for its prefix only.
            num_samples: How many suffixes to draw per prefix.
            sample: Whether to draw z from `p(z | prefix)`. False takes its mean instead,
                making the generation the model's single point prediction.
        Returns:
            The generated suffixes, `[batch_size, num_samples, steps]`, with row `(i, j)` the
            j-th sample for the i-th prefix of the batch.
        """
        prefix_pad_mask = item.prefix.pad_mask()  # [batch_size, seq_len]
        prefix = self.encoder(events=item.prefix, pad_mask=prefix_pad_mask)

        # Computed once per prefix: every sample of one prefix is drawn from the same
        # p(z | prefix), so running the prior once and repeating its parameters skips
        # num_samples - 1 redundant forward passes over identical rows.
        prior = self.prior(prefix.summary)

        # Repeat the prefix events and pad mask for every sample, so the decoder sees a batch of
        # size `batch_size * num_samples` and can generate all samples in one forward pass
        prefix_events = prefix.events.repeat_interleave(repeats=num_samples, dim=0)
        prefix_pad_mask = prefix_pad_mask.repeat_interleave(repeats=num_samples, dim=0)

        # One latent per sample: independent noise per draw, `Gaussian.sample()`'s
        # reparameterization trick, on top of the one prior distribution repeated per prefix.
        repeated = Gaussian(
            mean=prior.mean.repeat_interleave(repeats=num_samples, dim=0),
            logvar=prior.logvar.repeat_interleave(repeats=num_samples, dim=0),
        )
        z = repeated.sample() if sample else repeated.mean  # [rows, latent_dim]

        # A suffix holds at most `max_seq_len` events, the padded width the batch comes in at.
        # `sample=False` throughout: this decoder's heads are read at their mode whatever is
        # asked of them, the draw having already happened in z.
        generated = self.decoder.generate(
            z=z,
            prefix_encoded=prefix_events,
            prefix_pad_mask=prefix_pad_mask,
            max_steps=item.prefix.activities.size(dim=1),
            sample=False,
        )
        return self._per_sample(generated, batch_size=item.prefix.length.size(dim=0))


class Transformer(SuffixModel):
    """An encoder-decoder transformer that predicts a trace's suffix from its prefix, with no
    latent at all: the same backbone as `TransformerCVAE` with the latent path taken out.

    It is the arm the CVAE is read against, so it is the same everywhere the latent does not
    reach: one embedding space, the same encoder over the prefix, the same decoder
    cross-attending over its events. What differs is where the variability lives. With nothing
    conditioning the decoder, two runs of one prefix could only be the same suffix, so the draws
    have to come from the heads: the activity is sampled from its logits at every step, and both
    time heads emit a variance to be sampled from. That is exactly the per-step noise the latent
    exists to avoid, which is the comparison.

    Flow, with `prefix` and `suffix` both padded to `max_seq_len`:
        prefix                -> prefix events (for the decoder)
        prefix events, suffix -> an activity, the wait until it and a remaining time, at every
                                 suffix position
    """

    def __init__(self, config: TransformerConfig, codec: DatasetCodec):
        super().__init__(codec=codec)
        self.embeddings = EventEmbeddings(
            config=config.embeddings, codec=codec, d_model=config.d_model
        )
        # Only the prefix is ever read: with no posterior there is nothing a summary of the
        # ground-truth suffix would feed.
        self.encoder = TraceEncoder(
            config=config.encoder, embeddings=self.embeddings, d_model=config.d_model
        )
        self.decoder = Decoder(
            config=config.decoder,
            latent_config=None,
            embeddings=self.embeddings,
            d_model=config.d_model,
            num_activities=codec.activity.num_rows,
            sos_activity_index=codec.activity.sos_index,
            pad_activity_index=codec.activity.pad_index,
            pad_resource_index=codec.resource.pad_index,
            eot_activity_index=codec.activity.eot_index,
            stochastic=True,
        )

    def forward(self, item: SplitTrace) -> ModelOutput:
        """
        Args:
            item: A batch from `TraceDataset`, read for its prefix and for the suffix the
                decoder is teacher-forced on.
        Returns:
            The decoder's predictions, and no latents: there is no KL term to charge.
        """
        prefix_pad_mask = item.prefix.pad_mask()  # [batch_size, seq_len]
        prefix = self.encoder(events=item.prefix, pad_mask=prefix_pad_mask)

        decoder_output = self.decoder(
            suffix_activities=item.suffix.activities,
            z=None,
            prefix_encoded=prefix.events,
            prefix_pad_mask=prefix_pad_mask,
        )
        return ModelOutput(decoder=decoder_output, latents=None)

    @torch.no_grad()
    def generate(
        self, item: SplitTrace, *, num_samples: int, sample: bool = True
    ) -> GeneratedSuffix:
        """Generate `num_samples` suffixes for every prefix in `item`.

        Args:
            item: A batch from `TraceDataset`, read for its prefix only.
            num_samples: How many suffixes to draw per prefix. Every row decodes independently,
                so `num_samples` rows of one prefix give `num_samples` different suffixes.
            sample: Whether to sample the heads. False reads each at its mode - the likeliest
                activity, the mean of each time - which is the model's single point prediction,
                and makes every row of a prefix identical.
        Returns:
            The generated suffixes, `[batch_size, num_samples, steps]`, with row `(i, j)` the
            j-th sample for the i-th prefix of the batch.
        """
        prefix_pad_mask = item.prefix.pad_mask()  # [batch_size, seq_len]
        prefix = self.encoder(events=item.prefix, pad_mask=prefix_pad_mask)

        # The prefix is encoded once and repeated per sample, so the decoder writes every sample
        # of the batch in one pass. Unlike the CVAE nothing else is repeated: the noise that
        # separates two rows of a prefix is drawn inside the decode loop, step by step.
        prefix_events = prefix.events.repeat_interleave(repeats=num_samples, dim=0)
        prefix_pad_mask = prefix_pad_mask.repeat_interleave(repeats=num_samples, dim=0)

        generated = self.decoder.generate(
            z=None,
            prefix_encoded=prefix_events,
            prefix_pad_mask=prefix_pad_mask,
            max_steps=item.prefix.activities.size(dim=1),
            sample=sample,
        )
        return self._per_sample(generated, batch_size=item.prefix.length.size(dim=0))


def build_model(config: ModelConfig, codec: DatasetCodec) -> SuffixModel:
    """Build the architecture a config declares.

    Args:
        config: The run's `model` section, whose `kind` names the architecture.
        codec: The dataset the model is built against, supplying every data-derived dimension.
    Returns:
        The model, on the CPU and in training mode.
    """
    if isinstance(config, CVAEConfig):
        return TransformerCVAE(config=config, codec=codec)
    return Transformer(config=config, codec=codec)


def model_from_checkpoint(
    checkpoint: dict, codec: DatasetCodec, *, device: str = 'cpu'
) -> SuffixModel:
    """Rebuild the model a checkpoint holds, with its weights loaded.

    Which architecture that is comes out of the checkpoint's own config, so a checkpoint is read
    without being told what wrote it.

    Args:
        checkpoint: A checkpoint read by `load_checkpoint`.
        codec: The dataset the model is to be used on, supplying the vocabulary
            sizes and sequence length it was built against.
        device: Where to place the model.
    Returns:
        The model, in evaluation mode.
    Raises:
        ValueError: If the checkpoint does not carry a config and weights.
        pydantic.ValidationError: If the config it carries names no architecture, which is what
            a checkpoint written before `model.kind` existed looks like from here.
    """
    require_keys(checkpoint, MODEL_KEYS, purpose='rebuilt', remedy='Train the model again.')
    config = _MODEL_CONFIG.validate_python(checkpoint['model_config'])

    model = build_model(config=config, codec=codec).to(device=device)
    model.load_state_dict(state_dict=checkpoint['model_state_dict'])
    model.eval()
    return model
