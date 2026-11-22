from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import DiffusionConfig
from .latent import LatentStandardizer


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half_dim = dim // 2
    exponent = -math.log(10000.0) * torch.arange(
        half_dim,
        device=timesteps.device,
        dtype=torch.float32,
    ) / max(half_dim - 1, 1)
    embeddings = timesteps.float().unsqueeze(1) * torch.exp(exponent).unsqueeze(0)
    embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=1)
    if dim % 2 == 1:
        embeddings = F.pad(embeddings, (0, 1))
    return embeddings


def extract(values: torch.Tensor, timesteps: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    batch = timesteps.size(0)
    selected = values.gather(0, timesteps)
    return selected.view(batch, *([1] * (len(target_shape) - 1)))


class DiffusionSchedule(nn.Module):
    def __init__(
        self,
        config: DiffusionConfig,
        latent_standardizer: LatentStandardizer | None = None,
        sequence_length_counts: list[int] | None = None,
    ) -> None:
        super().__init__()
        if config.latent_dim is None or config.latent_dim <= 0:
            raise ValueError(
                "Diffusion latent_dim must be bound to the encoder embedding "
                "dimension before constructing the schedule."
            )
        if config.beta_schedule == "linear":
            betas = torch.linspace(config.beta_start, config.beta_end, config.num_steps)
        else:
            raise ValueError(f"Unsupported beta schedule: {config.beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("posterior_variance", posterior_variance.clamp_min(1e-8))
        self.register_buffer("posterior_guidance_variance", posterior_variance)
        self.num_steps = config.num_steps
        self.latent_standardizer = (
            latent_standardizer
            if latent_standardizer is not None
            else LatentStandardizer.identity(config.latent_dim)
        )
        if self.latent_standardizer.latent_dim != config.latent_dim:
            raise ValueError(
                f"Latent standardization dimension {self.latent_standardizer.latent_dim} "
                f"does not match diffusion dimension {config.latent_dim}."
            )
        if sequence_length_counts is None:
            sequence_length_counts = [0] * (config.max_sequence_length + 1)
            sequence_length_counts[config.max_sequence_length] = 1
        if len(sequence_length_counts) != config.max_sequence_length + 1:
            raise ValueError(
                "Sequence-length counts must contain one entry for every length from 0 "
                f"through {config.max_sequence_length}."
            )
        self.sequence_length_counts = [int(count) for count in sequence_length_counts]
        if sum(self.sequence_length_counts[1:]) <= 0:
            raise ValueError("Sequence-length distribution must contain at least one positive length.")

    def standardize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return self.latent_standardizer.standardize(latents)

    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return self.latent_standardizer.destandardize(latents)

    def sample_attention_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        weights = torch.tensor(self.sequence_length_counts, dtype=torch.float32, device=device)
        weights[0] = 0.0
        sampled_lengths = torch.multinomial(weights, num_samples=batch_size, replacement=True)
        positions = torch.arange(
            len(self.sequence_length_counts) - 1,
            device=device,
        ).unsqueeze(0)
        return positions < sampled_lengths.unsqueeze(1)

    def q_sample(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(latents)
        alpha_bar = extract(self.alphas_cumprod, timesteps, latents.shape)
        noisy = alpha_bar.sqrt() * latents + (1.0 - alpha_bar).sqrt() * noise
        return noisy, noise



def _modulate(inputs: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return inputs * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class _LatentDenoiserBase(nn.Module):
    """Shared masked epsilon-prediction objective for the retained denoiser."""
    def training_loss(
        self,
        clean_standardized_latents: torch.Tensor,
        attention_mask: torch.Tensor,
        schedule: DiffusionSchedule,
        *,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        return_predictions: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Compute masked DDPM noise loss over standardized token-level latents."""

        if clean_standardized_latents.ndim != 3:
            raise ValueError("Diffusion training requires [batch, sequence, features] latents.")
        attention_mask = attention_mask.to(dtype=torch.bool)
        valid_tokens = attention_mask.unsqueeze(-1)
        standardized_latents = clean_standardized_latents * valid_tokens.to(
            clean_standardized_latents.dtype
        )
        batch_size = standardized_latents.size(0)
        if timesteps is None:
            timesteps = torch.randint(
                low=0,
                high=schedule.num_steps,
                size=(batch_size,),
                device=clean_standardized_latents.device,
                dtype=torch.long,
            )
        elif timesteps.shape != (batch_size,):
            raise ValueError("Timesteps must contain one value per latent sequence.")
        if noise is None:
            noise = torch.randn_like(standardized_latents)
        elif noise.shape != standardized_latents.shape:
            raise ValueError("Diffusion noise must have the same shape as the latents.")
        noise = noise * valid_tokens.to(standardized_latents.dtype)
        noisy_latents, noise = schedule.q_sample(
            standardized_latents, timesteps, noise=noise
        )
        noisy_latents = noisy_latents * valid_tokens.to(noisy_latents.dtype)
        predicted_noise = self(noisy_latents, timesteps, attention_mask)
        squared_error = F.mse_loss(predicted_noise, noise, reduction="none")
        denominator = (
            valid_tokens.sum().clamp_min(1) * clean_standardized_latents.size(-1)
        )
        loss = (squared_error * valid_tokens).sum() / denominator
        outputs = {
            "loss": loss,
            "timesteps": timesteps,
        }
        if return_predictions:
            alpha_bar = extract(schedule.alphas_cumprod, timesteps, noisy_latents.shape)
            predicted_clean_latents = (
                noisy_latents - (1.0 - alpha_bar).sqrt() * predicted_noise
            ) / alpha_bar.sqrt().clamp_min(1e-8)
            predicted_clean_latents = predicted_clean_latents * valid_tokens.to(
                predicted_clean_latents.dtype
            )
            outputs.update(
                {
                    "noisy_latents": noisy_latents,
                    "predicted_noise": predicted_noise,
                    "predicted_clean_latents": predicted_clean_latents,
                }
            )
        return outputs


class FiLMResidualMLPMixerBlock(nn.Module):
    """Residual token/channel MLP block with AdaLN-Zero timestep conditioning."""

    def __init__(self, config: DiffusionConfig) -> None:
        super().__init__()
        hidden_dim = config.hidden_dim
        self.max_sequence_length = config.max_sequence_length
        self.norm_token = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.token_mlp = nn.Sequential(
            nn.Linear(config.max_sequence_length, config.mlp_token_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_token_hidden_dim, config.max_sequence_length),
        )
        self.norm_channel = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.channel_mlp = nn.Sequential(
            nn.Linear(hidden_dim, config.mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(config.dropout)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(
        self,
        hidden: torch.Tensor,
        time_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        (
            token_shift,
            token_scale,
            token_gate,
            channel_shift,
            channel_scale,
            channel_gate,
        ) = self.modulation(time_features).chunk(6, dim=-1)
        valid_tokens = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)

        token_input = _modulate(
            self.norm_token(hidden),
            token_shift,
            token_scale,
        ) * valid_tokens
        sequence_length = token_input.size(1)
        token_input = token_input.transpose(1, 2)
        if sequence_length < self.max_sequence_length:
            token_input = F.pad(
                token_input,
                (0, self.max_sequence_length - sequence_length),
            )
        token_output = self.token_mlp(token_input)[..., :sequence_length].transpose(1, 2)
        hidden = hidden + token_gate.unsqueeze(1) * self.dropout(token_output)
        hidden = hidden * valid_tokens

        channel_input = _modulate(
            self.norm_channel(hidden),
            channel_shift,
            channel_scale,
        ) * valid_tokens
        channel_output = self.channel_mlp(channel_input)
        hidden = hidden + channel_gate.unsqueeze(1) * self.dropout(channel_output)
        return hidden * valid_tokens


class ResidualMLPLatentDenoiser(_LatentDenoiserBase):
    """Sequence-aware residual MLP noise predictor for token-level latents."""

    def __init__(self, config: DiffusionConfig) -> None:
        super().__init__()
        if config.latent_dim is None or config.latent_dim <= 0:
            raise ValueError(
                "Denoiser latent_dim must be bound to the encoder embedding "
                "dimension before model construction."
            )
        if config.hidden_dim <= 0:
            raise ValueError("MLP denoiser hidden dimension must be positive.")
        if config.mlp_hidden_dim <= 0:
            raise ValueError("MLP denoiser channel hidden dimension must be positive.")
        if config.mlp_token_hidden_dim <= 0:
            raise ValueError("MLP denoiser token hidden dimension must be positive.")
        if config.denoiser_architecture != "mlp":
            raise ValueError("Only the residual MLP denoiser is supported.")
        if config.mlp_num_layers != 8:
            raise ValueError("The retained residual MLP denoiser must have 8 blocks.")

        self.config = config
        self.time_mlp = nn.Sequential(
            nn.Linear(config.time_embedding_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.input_projection = nn.Linear(config.latent_dim, config.hidden_dim)
        self.position_embedding = nn.Embedding(
            config.max_sequence_length,
            config.hidden_dim,
        )
        self.residual_layers = nn.ModuleList(
            [FiLMResidualMLPMixerBlock(config) for _ in range(config.mlp_num_layers)]
        )
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self.output_projection = nn.Linear(config.hidden_dim, config.latent_dim)

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_latents.ndim != 3:
            raise ValueError("MLP denoiser expects [batch, sequence, features] latents.")
        if attention_mask.shape != noisy_latents.shape[:2]:
            raise ValueError("Latent attention mask must match the batch and sequence dimensions.")
        if noisy_latents.size(1) > self.config.max_sequence_length:
            raise ValueError(
                f"Sequence length {noisy_latents.size(1)} exceeds configured maximum "
                f"{self.config.max_sequence_length}."
            )

        attention_mask = attention_mask.to(dtype=torch.bool)
        valid_tokens = attention_mask.unsqueeze(-1).to(dtype=noisy_latents.dtype)
        positions = torch.arange(noisy_latents.size(1), device=noisy_latents.device)
        hidden = self.input_projection(noisy_latents)
        hidden = hidden + self.position_embedding(positions).unsqueeze(0)
        hidden = hidden * valid_tokens
        time_features = self.time_mlp(
            timestep_embedding(timesteps, self.config.time_embedding_dim)
        )
        for block in self.residual_layers:
            hidden = block(hidden, time_features, attention_mask)
        predicted_noise = self.output_projection(self.output_norm(hidden))
        return predicted_noise * valid_tokens


DenoiserModel = ResidualMLPLatentDenoiser


def build_denoiser(config: DiffusionConfig) -> DenoiserModel:
    """Construct the sole retained eight-block residual MLP denoiser."""

    return ResidualMLPLatentDenoiser(config)
