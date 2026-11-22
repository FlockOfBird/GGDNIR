from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import DecoderConfig
from .tokenizer import SelfiesVocabulary


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_length, d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        return self.embedding(positions)


class EmbeddingToSelfiesDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        vocabulary: SelfiesVocabulary,
        config: DecoderConfig,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.vocabulary = vocabulary
        self.vocab_size = vocabulary.vocab_size
        self.config = config
        self.bos_id = vocabulary.bos_id
        self.eos_id = vocabulary.eos_id
        self.pad_id = vocabulary.pad_id

        self.token_embedding = nn.Embedding(self.vocab_size, config.d_model)
        self.position_embedding = PositionalEncoding(config.d_model, config.max_decode_length)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_layers)
        self.memory_projection = nn.Linear(latent_dim, config.d_model)
        self.memory_position_embedding = nn.Embedding(config.max_decode_length, config.d_model)
        self.output_projection = nn.Linear(config.d_model, self.vocab_size)
        self.dropout = nn.Dropout(config.dropout)

    def memory_from_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 3:
            raise ValueError("Decoder expects [batch, sequence, features] latent memory.")
        if latents.size(1) > self.config.max_decode_length:
            raise ValueError("Latent sequence exceeds the decoder's maximum length.")
        positions = torch.arange(latents.size(1), device=latents.device).unsqueeze(0)
        return self.memory_projection(latents) + self.memory_position_embedding(positions)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones((length, length), dtype=torch.bool, device=device), diagonal=1)

    def forward(
        self,
        latents: torch.Tensor,
        latent_attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.memory_from_latents(latents)
        if latent_attention_mask.shape != latents.shape[:2]:
            raise ValueError("Latent attention mask must match the decoder memory sequence.")
        token_features = self.token_embedding(decoder_input_ids) + self.position_embedding(decoder_input_ids)
        token_features = self.dropout(token_features)
        causal_mask = self._causal_mask(decoder_input_ids.size(1), decoder_input_ids.device)
        padding_mask = decoder_input_ids.eq(self.pad_id)
        decoded = self.decoder(
            tgt=token_features,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask,
            memory_key_padding_mask=~latent_attention_mask.to(dtype=torch.bool),
        )
        return self.output_projection(decoded)

    def reconstruction_loss(
        self,
        latents: torch.Tensor,
        latent_attention_mask: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Teacher-forced SELFIES loss for one latent representation.

        Joint training uses this entry point to decode the denoiser's
        differentiable clean-latent estimate without applying a second random
        corruption.
        """

        decoder_input_ids = target_ids[:, :-1]
        labels = target_ids[:, 1:]
        logits = self(latents, latent_attention_mask, decoder_input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            labels.reshape(-1),
            ignore_index=self.pad_id,
            label_smoothing=self.config.label_smoothing,
        )
        mask = labels.ne(self.pad_id)
        predictions = logits.argmax(dim=-1)
        token_accuracy = (
            (predictions.eq(labels) & mask).sum().float()
            / mask.sum().clamp_min(1).float()
        )
        return {
            "loss": loss,
            "token_accuracy": token_accuracy,
            "logits": logits,
        }

    def training_loss(
        self,
        latents: torch.Tensor,
        latent_attention_mask: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        decoder_input_ids = target_ids[:, :-1]
        labels = target_ids[:, 1:]

        clean_logits = self(latents, latent_attention_mask, decoder_input_ids)
        clean_loss = F.cross_entropy(
            clean_logits.reshape(-1, self.vocab_size),
            labels.reshape(-1),
            ignore_index=self.pad_id,
            label_smoothing=self.config.label_smoothing,
        )

        valid_tokens = latent_attention_mask.unsqueeze(-1).to(dtype=latents.dtype)
        latent_noise = torch.randn_like(latents) * self.config.latent_noise_std * valid_tokens
        robust_logits = self(latents + latent_noise, latent_attention_mask, decoder_input_ids)
        robust_loss = F.cross_entropy(
            robust_logits.reshape(-1, self.vocab_size),
            labels.reshape(-1),
            ignore_index=self.pad_id,
            label_smoothing=self.config.label_smoothing,
        )

        mask = labels.ne(self.pad_id)
        predictions = clean_logits.argmax(dim=-1)
        token_accuracy = (predictions.eq(labels) & mask).sum().float() / mask.sum().clamp_min(1).float()

        loss = clean_loss + self.config.robust_loss_weight * robust_loss
        return {
            "loss": loss,
            "clean_loss": clean_loss,
            "robust_loss": robust_loss,
            "token_accuracy": token_accuracy,
            "logits": clean_logits,
        }

    @torch.no_grad()
    def generate_token_ids(
        self,
        latents: torch.Tensor,
        latent_attention_mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        batch_size = latents.size(0)
        generated = torch.full(
            (batch_size, 1),
            self.bos_id,
            dtype=torch.long,
            device=latents.device,
        )

        for _ in range(self.config.max_decode_length - 1):
            logits = self(latents, latent_attention_mask, generated)
            next_token_logits = logits[:, -1] / max(temperature, 1e-6)
            probabilities = torch.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probabilities, num_samples=1)
            generated = torch.cat([generated, next_tokens], dim=1)
            if torch.all(next_tokens.squeeze(-1).eq(self.eos_id)):
                break

        return generated
