from __future__ import annotations

import torch
from tqdm import tqdm

from .decoder import EmbeddingToSelfiesDecoder
from .diffusion import DenoiserModel, DiffusionSchedule, extract
from .guidance import guidance_strength, noisy_latent_property_gradient
from .property_head import NoisyLatentPropertyRegressor
from .tokenizer import selfies_to_smiles


def reverse_diffusion_step(
    latents: torch.Tensor,
    attention_mask: torch.Tensor,
    step: int,
    denoiser: DenoiserModel,
    schedule: DiffusionSchedule,
    guidance_regressor: NoisyLatentPropertyRegressor | None = None,
    guidance_scale: float = 0.0,
    guidance_gamma: float = 1.0,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_size = latents.size(0)
    timesteps = torch.full((batch_size,), step, dtype=torch.long, device=latents.device)

    with torch.no_grad():
        predicted_noise = denoiser(latents, timesteps, attention_mask)
        alpha = extract(schedule.alphas, timesteps, latents.shape)
        alpha_bar = extract(schedule.alphas_cumprod, timesteps, latents.shape)
        beta = extract(schedule.betas, timesteps, latents.shape)
        mean = (latents - (beta / (1.0 - alpha_bar).sqrt()) * predicted_noise) / alpha.sqrt()

    variance = extract(schedule.posterior_variance, timesteps, latents.shape)
    guidance_variance = extract(schedule.posterior_guidance_variance, timesteps, latents.shape)

    if guidance_regressor is not None and guidance_scale != 0.0:
        scale = guidance_strength(
            step, schedule.num_steps, guidance_scale, guidance_gamma
        )
        if scale != 0.0:
            gradient = noisy_latent_property_gradient(
                latents,
                timesteps,
                guidance_regressor,
                attention_mask,
            )
            mean = mean + scale * guidance_variance * gradient

    valid_tokens = attention_mask.unsqueeze(-1).to(dtype=latents.dtype)
    if step == 0:
        return mean * valid_tokens

    with torch.no_grad():
        if noise is None:
            noise = torch.randn_like(latents)
        elif noise.shape != latents.shape or noise.device != latents.device or noise.dtype != latents.dtype:
            raise ValueError("Provided reverse-step noise must match latent shape, device, and dtype.")
        return (mean + variance.sqrt() * noise) * valid_tokens


def _decode_samples(
    token_ids: torch.Tensor,
    decoder: EmbeddingToSelfiesDecoder,
) -> list[dict[str, object]]:
    records = []
    for sample_token_ids in token_ids.cpu().tolist():
        selfies_text = decoder.vocabulary.decode(sample_token_ids)
        records.append(
            {
                "SMILES": selfies_to_smiles(selfies_text) or "",
                "SELFIES": selfies_text,
            }
        )
    return records


@torch.no_grad()
def sample_smiles(
    denoiser: DenoiserModel,
    schedule: DiffusionSchedule,
    decoder: EmbeddingToSelfiesDecoder,
    num_samples: int,
    batch_size: int,
    guidance_regressor: NoisyLatentPropertyRegressor | None = None,
    guidance_scale: float = 0.0,
    guidance_gamma: float = 1.0,
    temperature: float = 1.0,
    device: torch.device | None = None,
) -> list[dict[str, object]]:
    if device is None:
        device = next(denoiser.parameters()).device

    samples: list[dict[str, object]] = []
    denoiser.eval()
    decoder.eval()
    if guidance_regressor is not None:
        guidance_regressor.eval()

    with tqdm(total=num_samples, desc="sampling", unit="mol") as progress:
        while len(samples) < num_samples:
            current_batch_size = min(batch_size, num_samples - len(samples))
            attention_mask = schedule.sample_attention_mask(current_batch_size, device)
            latents = torch.randn(
                current_batch_size,
                denoiser.config.max_sequence_length,
                denoiser.config.latent_dim,
                device=device,
            )
            latents = latents * attention_mask.unsqueeze(-1).to(dtype=latents.dtype)
            for step in reversed(range(schedule.num_steps)):
                latents = reverse_diffusion_step(
                    latents=latents,
                    attention_mask=attention_mask,
                    step=step,
                    denoiser=denoiser,
                    schedule=schedule,
                    guidance_regressor=guidance_regressor,
                    guidance_scale=guidance_scale,
                    guidance_gamma=guidance_gamma,
                )

            token_ids = decoder.generate_token_ids(
                latents,
                attention_mask,
                temperature=temperature,
            )
            batch_records = _decode_samples(token_ids, decoder)
            samples.extend(batch_records)
            progress.update(len(batch_records))

    return samples
