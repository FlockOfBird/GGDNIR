from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModel, PreTrainedTokenizerBase

from .config import DataConfig, EncoderConfig
from .data import DatasetSplit
from .tokenizer import tokenize_smiles
from .utils import masked_mean


class PretrainedMoleculeEncoder(nn.Module):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.model = AutoModel.from_pretrained(
            config.model_name,
            local_files_only=config.local_files_only,
        )
        self.latent_dim = int(self.model.config.hidden_size)
        if self.model.config.model_type == "llama":
            # Token embedding extraction does not need a generation key/value cache.
            self.model.config.use_cache = False
            if config.pooling == "cls":
                raise ValueError("Llama has no bidirectional CLS embedding; use mean pooling.")
        if config.freeze_encoder:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

    def token_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

    def pool_embeddings(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.pooling == "cls":
            return hidden_states[:, 0]
        if self.config.pooling == "mean":
            return masked_mean(hidden_states, attention_mask)
        raise ValueError(f"Unsupported pooling strategy: {self.config.pooling}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = self.token_embeddings(input_ids, attention_mask)
        return self.pool_embeddings(hidden_states, attention_mask)


@dataclass
class EncodedSplit:
    smiles: list[str]
    labels: torch.Tensor | None
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    embeddings: torch.Tensor
    sequence_embeddings: torch.Tensor


def encode_dataframe(
    dataframe,
    tokenizer: PreTrainedTokenizerBase,
    encoder: PretrainedMoleculeEncoder,
    data_config: DataConfig,
    device: torch.device,
    *,
    pooled_only: bool = False,
) -> EncodedSplit:
    smiles = dataframe[data_config.smiles_column].astype(str).tolist()
    labels = None
    if data_config.label_column is not None:
        labels = torch.tensor(dataframe[data_config.label_column].astype(float).tolist(), dtype=torch.float32)
    pooled_cache = torch.empty((len(smiles), encoder.latent_dim), dtype=torch.float32)
    tokenized = None if pooled_only else tokenize_smiles(smiles, tokenizer, data_config)
    sequence_cache = None if pooled_only else torch.empty(
        (len(smiles), data_config.max_length, encoder.latent_dim), dtype=torch.float16
    )
    encoder.eval()

    batch_size = max(1, data_config.batch_size)
    with torch.no_grad():
        for start in range(0, len(smiles), batch_size):
            end = min(start + batch_size, len(smiles))
            if pooled_only and data_config.split_column is not None:
                # Benchmark input/target mismatch from silent tokenizer truncation
                # would change the published molecular distribution.
                full_tokens = tokenizer(
                    smiles[start:end], padding=False, truncation=False,
                    add_special_tokens=True,
                )
                if any(len(ids) > data_config.max_length for ids in full_tokens["input_ids"]):
                    raise ValueError(
                        f"Benchmark SMILES exceeds --max-length={data_config.max_length}; "
                        f"first overlength batch starts at molecule {start}. Increase --max-length."
                    )
            batch_tokens = (
                tokenize_smiles(smiles[start:end], tokenizer, data_config)
                if pooled_only else tokenized
            )
            assert batch_tokens is not None
            token_slice = slice(None) if pooled_only else slice(start, end)
            input_ids = batch_tokens["input_ids"][token_slice].to(device)
            attention_mask = batch_tokens["attention_mask"][token_slice].to(device)
            sequence_embeddings = encoder.token_embeddings(input_ids, attention_mask)
            embeddings = encoder.pool_embeddings(sequence_embeddings, attention_mask)
            pooled_cache[start:end].copy_(embeddings.to(dtype=torch.float32, device="cpu"))
            # Token-level caches dominate disk usage. Half precision is sufficient
            # for frozen encoder outputs and is converted back to float32 by trainers.
            if sequence_cache is not None:
                sequence_cache[start:end].copy_(
                    sequence_embeddings.to(dtype=torch.float16, device="cpu")
                )

    if pooled_only:
        # Downstream pooled models consume a length-one latent and mask. Avoid
        # retaining the much larger token IDs and token-level hidden states.
        return EncodedSplit(
            smiles=smiles,
            labels=labels,
            input_ids=torch.empty((len(smiles), 0), dtype=torch.long),
            attention_mask=torch.ones((len(smiles), 1), dtype=torch.bool),
            embeddings=pooled_cache,
            sequence_embeddings=pooled_cache.unsqueeze(1),
        )
    assert tokenized is not None and sequence_cache is not None

    return EncodedSplit(
        smiles=smiles,
        labels=labels,
        input_ids=tokenized["input_ids"],
        attention_mask=tokenized["attention_mask"],
        embeddings=pooled_cache,
        sequence_embeddings=sequence_cache,
    )


def encode_splits(
    splits: DatasetSplit,
    tokenizer: PreTrainedTokenizerBase,
    encoder: PretrainedMoleculeEncoder,
    data_config: DataConfig,
    device: torch.device,
    *,
    pooled_only: bool = False,
) -> dict[str, EncodedSplit]:
    return {
        "train": encode_dataframe(splits.train, tokenizer, encoder, data_config, device, pooled_only=pooled_only),
        "val": encode_dataframe(splits.val, tokenizer, encoder, data_config, device, pooled_only=pooled_only),
        "test": encode_dataframe(splits.test, tokenizer, encoder, data_config, device, pooled_only=pooled_only),
    }


# Preserve the old import name for existing callers.
PretrainedChemBERTaEncoder = PretrainedMoleculeEncoder
