from __future__ import annotations

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from src.datasets.codec import DatasetCodec
from src.datasets.dataset import SplitTrace
from src.model.components.decoder import GeneratedSuffix
from src.model.components.diffusion import CosineSchedule, DiffusionDenoiser
from src.model.components.embeddings import EventEmbeddings
from src.model.components.trace_encoder import TraceEncoder
from src.model.models import DiffusionOutput, ModelOutput, SuffixModel
from src.training import LatentMetrics, Loss


class MaskedDiffusionTransformer(SuffixModel):
    """Length-first conditional diffusion model for complete suffixes and their times."""

    def __init__(self, config: DictConfig, codec: DatasetCodec):
        super().__init__(codec=codec)
        self.max_length = codec.max_trace_length
        self.mask_activity_index = codec.activity.num_rows
        self.sample_steps = config.diffusion.sample_steps
        self.ddim_eta = config.diffusion.ddim_eta
        self.continuous_clip = config.diffusion.continuous_clip
        self.loss_config = config.loss
        self.fallback_activity_index = codec.activity.unk_index

        self.embeddings = EventEmbeddings(
            config=config.embeddings,
            codec=codec,
            d_model=config.d_model,
            extra_activity_rows=1,
        )
        self.encoder = TraceEncoder(
            config=config.encoder, embeddings=self.embeddings, d_model=config.d_model
        )
        self.length_head = nn.Sequential(
            nn.Linear(in_features=config.d_model, out_features=config.decoder.head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=config.decoder.dropout),
            nn.Linear(in_features=config.decoder.head_hidden_dim, out_features=self.max_length),
        )
        self.denoiser = DiffusionDenoiser(
            config=config.decoder,
            embeddings=self.embeddings,
            d_model=config.d_model,
            max_length=self.max_length,
            num_activities=codec.activity.num_rows,
            pad_resource_index=codec.resource.pad_index,
            diffusion_steps=config.diffusion.train_steps,
        )
        self.schedule = CosineSchedule(
            steps=config.diffusion.train_steps, offset=config.diffusion.cosine_offset
        )
        self.register_buffer(
            name='unemittable_activities',
            tensor=torch.tensor(
                data=[codec.activity.pad_index, codec.activity.eot_index, codec.activity.sos_index],
                dtype=torch.long,
            ),
            persistent=False,
        )

    def _length_logits(
        self, prefix_summary: torch.Tensor, prefix_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Predict content length while excluding lengths that cannot fit after the prefix."""
        logits = self.length_head(prefix_summary)  # [batch_size, max_length]
        lengths = torch.arange(
            start=1, end=self.max_length + 1, device=prefix_lengths.device
        )  # [max_length]
        maximum = (self.max_length - prefix_lengths).clamp(min=1)
        invalid = lengths.unsqueeze(dim=0) > maximum.unsqueeze(dim=1)
        return logits.masked_fill(mask=invalid, value=-torch.inf)

    @staticmethod
    def _valid_positions(lengths: torch.Tensor, *, width: int) -> torch.Tensor:
        positions = torch.arange(end=width, device=lengths.device)
        return positions.unsqueeze(dim=0) < lengths.unsqueeze(dim=1)

    def _activity_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.index_fill(dim=-1, index=self.unemittable_activities, value=-torch.inf)

    def _corrupt_activities(
        self,
        activities: torch.Tensor,
        *,
        valid: torch.Tensor,
        alpha_bars: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask_probability = 1.0 - alpha_bars
        masked = (torch.rand_like(activities, dtype=torch.float32) < mask_probability) & valid

        # Every trace contributes an activity target, including the very low-noise timesteps.
        missing = ~masked.any(dim=1)
        if bool(missing.any()):
            choices = torch.rand_like(activities, dtype=torch.float32).masked_fill(
                mask=~valid, value=-1.0
            )
            forced = choices.argmax(dim=1)
            masked[missing, forced[missing]] = True

        corrupted = activities.masked_fill(mask=masked, value=self.mask_activity_index)
        corrupted = corrupted.masked_fill(mask=~valid, value=self.pad_activity_index)
        return corrupted, masked

    def forward(self, item: SplitTrace) -> ModelOutput:
        prefix_pad_mask = item.prefix.pad_mask()
        prefix = self.encoder(events=item.prefix, pad_mask=prefix_pad_mask)
        lengths = item.suffix.length - 1
        valid = self._valid_positions(lengths, width=item.suffix.activities.size(dim=1))

        timesteps = torch.randint(
            low=1,
            high=self.schedule.steps + 1,
            size=(lengths.size(dim=0),),
            device=lengths.device,
        )
        alpha_bars = self.schedule.at(timesteps, like=item.inter_event_times)
        activities, activity_mask = self._corrupt_activities(
            activities=item.suffix.activities, valid=valid, alpha_bars=alpha_bars
        )

        inter_event_noise = torch.randn_like(input=item.inter_event_times)
        inter_event_times = (
            alpha_bars.sqrt() * item.inter_event_times
            + (1.0 - alpha_bars).sqrt() * inter_event_noise
        ).masked_fill(mask=~valid, value=0.0)

        remaining_target = item.remaining_times[:, 0]
        remaining_noise = torch.randn_like(input=remaining_target)
        remaining_alpha = alpha_bars[:, 0]
        remaining_times = (
            remaining_alpha.sqrt() * remaining_target
            + (1.0 - remaining_alpha).sqrt() * remaining_noise
        )

        denoising = self.denoiser(
            activities=activities,
            inter_event_times=inter_event_times,
            remaining_times=remaining_times,
            lengths=lengths,
            timesteps=timesteps,
            prefix_encoded=prefix.events,
            prefix_pad_mask=prefix_pad_mask,
        )
        return ModelOutput(
            decoder=None,
            latents=None,
            diffusion=DiffusionOutput(
                denoising=denoising,
                length_logits=self._length_logits(prefix.summary, item.prefix.length),
                activity_targets=item.suffix.activities,
                activity_mask=activity_mask,
                valid_positions=valid,
                inter_event_noise=inter_event_noise,
                remaining_time_noise=remaining_noise,
            ),
        )

    def compute_loss(
        self, output: ModelOutput, batch: SplitTrace, *, step: int
    ) -> tuple[torch.Tensor, Loss, LatentMetrics | None]:
        """Score length prediction and all three denoising targets per trace."""
        del step
        if output.diffusion is None:
            raise ValueError('MaskedDiffusionTransformer requires a diffusion output')
        predicted = output.diffusion
        batch_size = batch.suffix.length.size(dim=0)
        lengths = batch.suffix.length - 1

        length_loss = F.cross_entropy(
            input=predicted.length_logits, target=lengths - 1, reduction='sum'
        )

        activity_targets = predicted.activity_targets.masked_fill(
            mask=~predicted.activity_mask, value=self.fallback_activity_index
        )
        activity_losses = F.cross_entropy(
            input=self._activity_logits(predicted.denoising.activity_logits).transpose(1, 2),
            target=activity_targets,
            reduction='none',
        )
        activity_counts = predicted.activity_mask.sum(dim=1).clamp(min=1)
        activity_loss = (
            activity_losses.masked_fill(mask=~predicted.activity_mask, value=0.0).sum(dim=1)
            / activity_counts
        ).sum()

        inter_event_losses = F.mse_loss(
            input=predicted.denoising.inter_event_noise,
            target=predicted.inter_event_noise,
            reduction='none',
        )
        valid_counts = predicted.valid_positions.sum(dim=1).clamp(min=1)
        inter_event_time_loss = (
            inter_event_losses.masked_fill(mask=~predicted.valid_positions, value=0.0).sum(dim=1)
            / valid_counts
        ).sum()
        remaining_time_loss = F.mse_loss(
            input=predicted.denoising.remaining_time_noise,
            target=predicted.remaining_time_noise,
            reduction='sum',
        )

        reconstruction_loss = (
            self.loss_config.activity_weight * activity_loss
            + self.loss_config.inter_event_time_weight * inter_event_time_loss
            + self.loss_config.remaining_time_weight * remaining_time_loss
        )
        total_loss = self.loss_config.length_weight * length_loss + reconstruction_loss
        metrics = Loss(
            loss=total_loss.item(),
            reconstruction_loss=reconstruction_loss.item(),
            length_loss=length_loss.item(),
            activity_loss=activity_loss.item(),
            inter_event_time_loss=inter_event_time_loss.item(),
            remaining_time_loss=remaining_time_loss.item(),
        )
        return total_loss / batch_size, metrics, None

    def _sample_lengths(
        self,
        logits: torch.Tensor,
        *,
        num_samples: int,
        sample: bool,
    ) -> torch.Tensor:
        if sample:
            return torch.multinomial(
                input=logits.softmax(dim=-1), num_samples=num_samples, replacement=True
            ).add(1)
        point = logits.argmax(dim=-1, keepdim=True).add(1)
        return point.expand(-1, num_samples)

    @staticmethod
    def _deterministic_reveal(
        probabilities: torch.Tensor,
        *,
        masked: torch.Tensor,
        valid: torch.Tensor,
        lengths: torch.Tensor,
        alpha_previous: torch.Tensor,
    ) -> torch.Tensor:
        confidence = probabilities.max(dim=-1).values.masked_fill(mask=~masked, value=-torch.inf)
        order = confidence.argsort(dim=1, descending=True)
        ranks = torch.empty_like(input=order)
        ranks.scatter_(
            dim=1,
            index=order,
            src=torch.arange(end=order.size(dim=1), device=order.device)
            .unsqueeze(dim=0)
            .expand_as(order),
        )
        desired = (alpha_previous * lengths).round().to(torch.long)
        current = ((~masked) & valid).sum(dim=1)
        count = (desired - current).clamp(min=0)
        return masked & (ranks < count.unsqueeze(dim=1))

    def _reverse_activities(
        self,
        activities: torch.Tensor,
        logits: torch.Tensor,
        *,
        valid: torch.Tensor,
        lengths: torch.Tensor,
        timestep: int,
        previous: int,
        sample: bool,
    ) -> torch.Tensor:
        probabilities = self._activity_logits(logits).softmax(dim=-1)
        if sample:
            candidates = torch.multinomial(
                input=probabilities.flatten(end_dim=1), num_samples=1
            ).view_as(activities)
        else:
            candidates = probabilities.argmax(dim=-1)

        masked = (activities == self.mask_activity_index) & valid
        alpha_t = self.schedule.alpha_bars[timestep]
        alpha_previous = self.schedule.alpha_bars[previous]
        if sample:
            reveal_probability = ((alpha_previous - alpha_t) / (1.0 - alpha_t)).clamp(0.0, 1.0)
            reveal = masked & (
                torch.rand_like(activities, dtype=torch.float32) < reveal_probability
            )
        else:
            reveal = self._deterministic_reveal(
                probabilities=probabilities,
                masked=masked,
                valid=valid,
                lengths=lengths,
                alpha_previous=alpha_previous,
            )
        return torch.where(condition=reveal, input=candidates, other=activities)

    def _reverse_continuous(
        self,
        values: torch.Tensor,
        noise: torch.Tensor,
        *,
        timestep: int,
        previous: int,
        sample: bool,
    ) -> torch.Tensor:
        alpha_t = self.schedule.alpha_bars[timestep].to(dtype=values.dtype)
        alpha_previous = self.schedule.alpha_bars[previous].to(dtype=values.dtype)
        clean = ((values - (1.0 - alpha_t).sqrt() * noise) / alpha_t.sqrt()).clamp(
            min=-self.continuous_clip, max=self.continuous_clip
        )
        eta = self.ddim_eta if sample else 0.0
        sigma = (
            eta
            * (((1.0 - alpha_previous) / (1.0 - alpha_t)) * (1.0 - alpha_t / alpha_previous))
            .clamp(min=0.0)
            .sqrt()
        )
        direction = (1.0 - alpha_previous - sigma.square()).clamp(min=0.0).sqrt() * noise
        random = torch.randn_like(input=values) if sample and previous > 0 else 0.0
        return alpha_previous.sqrt() * clean + direction + sigma * random

    @torch.no_grad()
    def generate(
        self, item: SplitTrace, *, num_samples: int, sample: bool = True
    ) -> GeneratedSuffix:
        """Predict lengths first, then denoise every position of every suffix in parallel."""
        prefix_pad_mask = item.prefix.pad_mask()
        prefix = self.encoder(events=item.prefix, pad_mask=prefix_pad_mask)
        length_logits = self._length_logits(prefix.summary, item.prefix.length)
        lengths = self._sample_lengths(
            logits=length_logits, num_samples=num_samples, sample=sample
        )  # [batch_size, num_samples]

        batch_size = item.prefix.length.size(dim=0)
        flat_lengths = lengths.flatten()  # [batch_size * num_samples]
        width = int(flat_lengths.max().item())
        valid = self._valid_positions(flat_lengths, width=width)
        rows = flat_lengths.size(dim=0)
        prefix_events = prefix.events.repeat_interleave(repeats=num_samples, dim=0)
        repeated_prefix_mask = prefix_pad_mask.repeat_interleave(repeats=num_samples, dim=0)

        activities = torch.full(
            size=(rows, width),
            fill_value=self.mask_activity_index,
            dtype=torch.long,
            device=flat_lengths.device,
        )
        inter_event_times = (
            torch.randn(size=(rows, width), device=flat_lengths.device)
            if sample
            else torch.zeros(size=(rows, width), device=flat_lengths.device)
        )
        remaining_times = (
            torch.randn(size=(rows,), device=flat_lengths.device)
            if sample
            else torch.zeros(size=(rows,), device=flat_lengths.device)
        )

        timesteps = self.schedule.sampling_timesteps(self.sample_steps)
        for position, timestep in enumerate(timesteps):
            previous = timesteps[position + 1] if position + 1 < len(timesteps) else 0
            step = torch.full(
                size=(rows,), fill_value=timestep, dtype=torch.long, device=flat_lengths.device
            )
            prediction = self.denoiser(
                activities=activities,
                inter_event_times=inter_event_times,
                remaining_times=remaining_times,
                lengths=flat_lengths,
                timesteps=step,
                prefix_encoded=prefix_events,
                prefix_pad_mask=repeated_prefix_mask,
            )
            activities = self._reverse_activities(
                activities=activities,
                logits=prediction.activity_logits,
                valid=valid,
                lengths=flat_lengths,
                timestep=timestep,
                previous=previous,
                sample=sample,
            )
            inter_event_times = self._reverse_continuous(
                values=inter_event_times,
                noise=prediction.inter_event_noise,
                timestep=timestep,
                previous=previous,
                sample=sample,
            ).masked_fill(mask=~valid, value=0.0)
            remaining_times = self._reverse_continuous(
                values=remaining_times,
                noise=prediction.remaining_time_noise,
                timestep=timestep,
                previous=previous,
                sample=sample,
            )

        activities = activities.masked_fill(mask=~valid, value=self.pad_activity_index)
        generated = GeneratedSuffix(
            activities=activities,
            lengths=flat_lengths,
            inter_event_times=inter_event_times,
            remaining_time=remaining_times,
        )
        return self._per_sample(generated, batch_size=batch_size)
