import torch
import torch.nn.functional as F

from src.configs.schema import TransformerConfig
from src.datasets.codec import DatasetCodec
from src.datasets.dataset import SplitTrace
from src.distributions.gaussian import Gaussian
from src.model.components.decoder import Decoder, GeneratedSuffix
from src.model.components.embeddings import EventEmbeddings
from src.model.components.trace_encoder import TraceEncoder
from src.model.models import ModelOutput, SuffixModel, _timed_positions
from src.training.kl import LatentMetrics
from src.training.loss import Loss


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

    def compute_loss(
        self, output: ModelOutput, batch: SplitTrace, *, step: int
    ) -> tuple[torch.Tensor, Loss, LatentMetrics | None]:
        """Score a forward pass by its reconstruction alone: no latent, no KL term to charge.

        Both time heads are genuinely stochastic here (`Decoder(..., stochastic=True)`), sampled
        from at generation, so they are scored by their Gaussian NLL rather than a plain error:
        a head that is wrong pays less by widening, which is the price of being drawn from.
        """
        batch_size = batch.suffix.activities.size(0)

        activity_loss = F.cross_entropy(
            output.decoder.activity_logits.transpose(1, 2),
            batch.suffix.activities,
            ignore_index=self.pad_activity_index,
            reduction='sum',
        )

        timed = _timed_positions(batch)  # [batch_size, seq_len]
        time_to_next_loss = _gaussian_nll(
            prediction=output.decoder.times_to_next, target=batch.times_to_next, mask=timed
        )
        remaining_time_loss = _gaussian_nll(
            prediction=output.decoder.remaining_times, target=batch.remaining_times, mask=timed
        )

        reconstruction_loss = activity_loss + time_to_next_loss + remaining_time_loss

        metrics = Loss(
            loss=reconstruction_loss.item(),
            reconstruction_loss=reconstruction_loss.item(),
            activity_loss=activity_loss.item(),
            time_to_next_loss=time_to_next_loss.item(),
            remaining_time_loss=remaining_time_loss.item(),
        )
        return reconstruction_loss / batch_size, metrics, None


def _gaussian_nll(
    *, prediction: Gaussian, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """The negative log-likelihood of one time head, over the positions its target is defined at,
    without the constant the two halves of a comparison would both carry.

    A head that emits its own scale can buy its way out of a hard position by widening, which is
    the price of letting it be sampled from: a model whose draws come from its heads has to say
    how wide they are.

    Args:
        prediction: The head's distribution, `[batch_size, seq_len]` per field, standardized.
        target: What it is scored against, the shape and scale of `prediction.mean`.
        mask: True at the positions to score.
    Returns:
        A scalar, summed over the scored positions of the whole batch.
    """
    # exp(-logvar) rather than a division by the variance: one exp and a multiply, and no
    # reciprocal of a number that a wide head can drive towards 0.
    error = 0.5 * (
        prediction.logvar
        + (prediction.mean - target).pow(exponent=2) * torch.exp(input=-prediction.logvar)
    )  # [batch_size, seq_len]
    return error.masked_fill(mask=~mask, value=0.0).sum()
