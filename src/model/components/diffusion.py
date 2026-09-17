from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from omegaconf import DictConfig
from torch import nn

from src.datasets.dataset import Events
from src.model.components.attention import MultiHeadAttention
from src.model.components.conditioning import AdaLNConditioning, LayerModulation, gate, modulate
from src.model.components.embeddings import EventEmbeddings


@dataclass(frozen=True)
class DenoisingOutput:
    """Clean activity logits and continuous noise estimates at one diffusion step."""

    activity_logits: torch.Tensor  # [batch_size, seq_len, num_activities]
    inter_event_noise: torch.Tensor  # [batch_size, seq_len]
    remaining_time_noise: torch.Tensor  # [batch_size]


class CosineSchedule(nn.Module):
    """Cosine cumulative signal schedule shared by discrete and continuous corruption."""

    def __init__(self, *, steps: int, offset: float):
        super().__init__()
        positions = torch.linspace(start=0.0, end=1.0, steps=steps + 1)
        angles = ((positions + offset) / (1.0 + offset)) * math.pi / 2.0
        alpha_bars = torch.cos(angles).square()
        alpha_bars = alpha_bars / alpha_bars[0]
        # The final value must be positive for stable divisions in the reverse update.
        alpha_bars = alpha_bars.clamp(min=1e-5, max=1.0)
        self.steps = steps
        self.register_buffer(name='alpha_bars', tensor=alpha_bars)

    def at(self, timesteps: torch.Tensor, *, like: torch.Tensor) -> torch.Tensor:
        """Read cumulative signal at integer timesteps and match a target tensor's rank."""
        values = self.alpha_bars[timesteps].to(dtype=like.dtype)  # [batch_size]
        return values.view(values.size(dim=0), *((1,) * (like.ndim - 1)))

    def sampling_timesteps(self, count: int) -> tuple[int, ...]:
        """Return a descending, unique set of reverse-process timesteps."""
        selected = torch.linspace(start=self.steps, end=1, steps=count).round().to(torch.long)
        return tuple(int(step) for step in torch.unique_consecutive(selected))


class DenoisingLayer(nn.Module):
    """Bidirectional suffix self-attention followed by prefix cross-attention."""

    def __init__(self, config: DictConfig, *, d_model: int):
        super().__init__()
        self.self_attention = MultiHeadAttention(
            d_model=d_model, num_heads=config.num_heads, dropout=config.dropout
        )
        self.cross_attention = MultiHeadAttention(
            d_model=d_model, num_heads=config.num_heads, dropout=config.dropout
        )
        self.feedforward = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=config.feedforward_dim),
            nn.ReLU(),
            nn.Dropout(p=config.dropout),
            nn.Linear(in_features=config.feedforward_dim, out_features=d_model),
        )
        self.self_attention_norm = nn.LayerNorm(normalized_shape=d_model, elementwise_affine=False)
        self.cross_attention_norm = nn.LayerNorm(normalized_shape=d_model, elementwise_affine=False)
        self.feedforward_norm = nn.LayerNorm(normalized_shape=d_model, elementwise_affine=False)
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        suffix_pad_mask: torch.Tensor,
        prefix_encoded: torch.Tensor,
        prefix_pad_mask: torch.Tensor,
        modulation: LayerModulation,
    ) -> torch.Tensor:
        """Denoise all suffix positions together while conditioning on the prefix."""
        hidden_norm = modulate(
            hidden=self.self_attention_norm(hidden), modulation=modulation.self_attention
        )
        hidden = hidden + gate(
            branch=self.dropout(
                self.self_attention(
                    query=hidden_norm,
                    keys_values=self.self_attention.project(hidden_norm),
                    key_padding_mask=suffix_pad_mask,
                )
            ),
            modulation=modulation.self_attention,
        )
        hidden = hidden + gate(
            branch=self.dropout(
                self.cross_attention(
                    query=modulate(
                        hidden=self.cross_attention_norm(hidden),
                        modulation=modulation.cross_attention,
                    ),
                    keys_values=self.cross_attention.project(prefix_encoded),
                    key_padding_mask=prefix_pad_mask,
                )
            ),
            modulation=modulation.cross_attention,
        )
        return hidden + gate(
            branch=self.dropout(
                self.feedforward(
                    modulate(
                        hidden=self.feedforward_norm(hidden),
                        modulation=modulation.feedforward,
                    )
                )
            ),
            modulation=modulation.feedforward,
        )


class DiffusionDenoiser(nn.Module):
    """Transformer denoiser over a fixed-length suffix and one remaining-time token."""

    def __init__(
        self,
        config: DictConfig,
        *,
        embeddings: EventEmbeddings,
        d_model: int,
        max_length: int,
        num_activities: int,
        pad_resource_index: int,
        diffusion_steps: int,
    ):
        super().__init__()
        self.embeddings = embeddings
        self.pad_resource_index = pad_resource_index
        self.diffusion_steps = diffusion_steps
        self.max_length = max_length

        self.embedding_norm = nn.LayerNorm(normalized_shape=d_model)
        self.dropout = nn.Dropout(p=config.dropout)
        self.remaining_time_token = nn.Parameter(data=torch.zeros(size=(1, 1, d_model)))
        self.remaining_time_projection = nn.Linear(in_features=1, out_features=d_model)
        self.length_embedding = nn.Embedding(num_embeddings=max_length + 1, embedding_dim=d_model)
        self.timestep_projection = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_model),
            nn.SiLU(),
            nn.Linear(in_features=d_model, out_features=d_model),
        )
        self.conditioning = AdaLNConditioning(
            latent_dim=d_model, d_model=d_model, num_layers=config.num_layers
        )
        self.layers = nn.ModuleList(
            DenoisingLayer(config, d_model=d_model) for _ in range(config.num_layers)
        )
        self.norm = nn.LayerNorm(normalized_shape=d_model)
        self.shared_layer = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=config.head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=config.dropout),
        )
        self.activity_head = nn.Linear(
            in_features=config.head_hidden_dim, out_features=num_activities
        )
        self.inter_event_time_head = nn.Linear(in_features=config.head_hidden_dim, out_features=1)
        self.remaining_time_head = nn.Linear(in_features=config.head_hidden_dim, out_features=1)

    def forward(
        self,
        *,
        activities: torch.Tensor,
        inter_event_times: torch.Tensor,
        remaining_times: torch.Tensor,
        lengths: torch.Tensor,
        timesteps: torch.Tensor,
        prefix_encoded: torch.Tensor,
        prefix_pad_mask: torch.Tensor,
    ) -> DenoisingOutput:
        """Predict clean activities and continuous noise from a corrupted suffix."""
        content = self.embeddings(
            self._events(activities, inter_event_times)
        )  # [batch_size, seq_len, d_model]
        remaining = self.remaining_time_token.expand(content.size(dim=0), -1, -1)
        remaining = remaining + self.remaining_time_projection(
            remaining_times[:, None, None]
        )  # [batch_size, 1, d_model]
        hidden = self.embedding_norm(
            self.dropout(torch.cat(tensors=(remaining, content), dim=1))
        )  # [batch_size, 1 + seq_len, d_model]

        positions = torch.arange(end=activities.size(dim=1), device=activities.device)
        content_pad_mask = positions.unsqueeze(dim=0) >= lengths.unsqueeze(dim=1)
        suffix_pad_mask = torch.cat(
            tensors=(content_pad_mask.new_zeros(size=(content.size(dim=0), 1)), content_pad_mask),
            dim=1,
        )  # [batch_size, 1 + seq_len]

        conditioning = self.timestep_projection(self._timestep_embedding(timesteps))
        conditioning = conditioning + self.length_embedding(lengths)
        modulations = self.conditioning.layers(conditioning)
        for layer, modulation in zip(self.layers, modulations, strict=True):
            hidden = layer(
                hidden,
                suffix_pad_mask=suffix_pad_mask,
                prefix_encoded=prefix_encoded,
                prefix_pad_mask=prefix_pad_mask,
                modulation=modulation,
            )
        hidden = self.norm(hidden)

        remaining_features = self.shared_layer(hidden[:, 0])
        content_features = self.shared_layer(hidden[:, 1:])
        return DenoisingOutput(
            activity_logits=self.activity_head(content_features),
            inter_event_noise=self.inter_event_time_head(content_features).squeeze(dim=-1),
            remaining_time_noise=self.remaining_time_head(remaining_features).squeeze(dim=-1),
        )

    def _timestep_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Build a sinusoidal embedding of normalized integer diffusion steps."""
        dimension = self.length_embedding.embedding_dim
        half = dimension // 2
        frequencies = torch.exp(
            input=torch.arange(end=half, device=timesteps.device, dtype=torch.float32)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        angles = timesteps.float().unsqueeze(dim=1) * frequencies
        embedded = torch.cat(tensors=(angles.sin(), angles.cos()), dim=1)
        if dimension % 2:
            embedded = torch.nn.functional.pad(input=embedded, pad=(0, 1))
        return embedded

    def _events(self, activities: torch.Tensor, inter_event_times: torch.Tensor) -> Events:
        """Wrap noisy suffix channels in the event structure shared embeddings consume."""
        batch_size, seq_len = activities.shape
        device = activities.device
        return Events(
            activities=activities,
            resources=torch.full(
                size=(batch_size, seq_len),
                fill_value=self.pad_resource_index,
                dtype=torch.long,
                device=device,
            ),
            inter_event_times=inter_event_times,
            categorical_attributes=torch.zeros(
                size=(batch_size, seq_len, self.embeddings.num_categorical),
                dtype=torch.long,
                device=device,
            ),
            numeric_attributes=torch.zeros(
                size=(batch_size, seq_len, self.embeddings.num_numeric), device=device
            ),
            numeric_attributes_present=torch.zeros(
                size=(batch_size, seq_len, self.embeddings.num_numeric), device=device
            ),
            length=torch.full(
                size=(batch_size,), fill_value=seq_len, dtype=torch.long, device=device
            ),
        )
