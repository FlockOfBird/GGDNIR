from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import selfies as sf
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .config import DataConfig, EncoderConfig


def load_encoder_tokenizer(
    encoder_config: EncoderConfig,
) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(
        encoder_config.model_name,
        use_fast=True,
        local_files_only=encoder_config.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token or tokenizer.cls_token
    return tokenizer


# Preserve the old import name for existing callers.
load_chemberta_tokenizer = load_encoder_tokenizer


def tokenize_smiles(
    smiles: list[str],
    tokenizer: PreTrainedTokenizerBase,
    data_config: DataConfig,
) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        smiles,
        padding="max_length",
        truncation=True,
        max_length=data_config.max_length,
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }


def smiles_to_selfies(smiles: str) -> str | None:
    try:
        return sf.encoder(smiles)
    except Exception:
        return None


def selfies_to_smiles(selfies_text: str) -> str | None:
    try:
        smiles = sf.decoder(selfies_text)
    except Exception:
        return None
    smiles = smiles.strip()
    return smiles or None


def split_selfies_tokens(selfies_text: str) -> list[str]:
    return list(sf.split_selfies(selfies_text))


@dataclass(frozen=True)
class SelfiesSpecialTokens:
    pad: str = "<pad>"
    bos: str = "<bos>"
    eos: str = "<eos>"
    unk: str = "<unk>"


class SelfiesVocabulary:
    def __init__(self, token_to_id: dict[str, int], max_length: int) -> None:
        self.token_to_id = token_to_id
        self.id_to_token = {idx: token for token, idx in token_to_id.items()}
        self.max_length = max_length
        self.special_tokens = SelfiesSpecialTokens()
        self.pad_id = token_to_id[self.special_tokens.pad]
        self.bos_id = token_to_id[self.special_tokens.bos]
        self.eos_id = token_to_id[self.special_tokens.eos]
        self.unk_id = token_to_id[self.special_tokens.unk]

    @classmethod
    def build(cls, selfies_list: list[str], max_length: int) -> "SelfiesVocabulary":
        counter = Counter()
        for selfies_text in selfies_list:
            counter.update(split_selfies_tokens(selfies_text))

        specials = SelfiesSpecialTokens()
        tokens = [specials.pad, specials.bos, specials.eos, specials.unk]
        tokens.extend(token for token, _count in sorted(counter.items()))
        token_to_id = {token: idx for idx, token in enumerate(tokens)}
        return cls(token_to_id=token_to_id, max_length=max_length)

    @classmethod
    def from_tokens(cls, tokens: list[str], max_length: int) -> "SelfiesVocabulary":
        return cls({token: idx for idx, token in enumerate(tokens)}, max_length=max_length)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def ordered_tokens(self) -> list[str]:
        return [self.id_to_token[idx] for idx in range(len(self.id_to_token))]

    def encode(self, selfies_text: str) -> list[int]:
        tokens = split_selfies_tokens(selfies_text)
        max_body_length = self.max_length - 2
        body = [self.token_to_id.get(token, self.unk_id) for token in tokens[:max_body_length]]
        ids = [self.bos_id, *body, self.eos_id]
        padding = [self.pad_id] * max(0, self.max_length - len(ids))
        return ids + padding

    def decode(self, token_ids: list[int]) -> str:
        tokens: list[str] = []
        for token_id in token_ids:
            if token_id == self.eos_id:
                break
            if token_id in (self.pad_id, self.bos_id):
                continue
            tokens.append(self.id_to_token.get(token_id, self.special_tokens.unk))
        return "".join(tokens)


def encode_selfies_batch(
    selfies_list: list[str],
    vocabulary: SelfiesVocabulary,
) -> torch.Tensor:
    return torch.tensor([vocabulary.encode(selfies_text) for selfies_text in selfies_list], dtype=torch.long)
