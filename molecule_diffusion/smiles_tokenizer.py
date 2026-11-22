"""Serializable atom-level SMILES tokenizer for small-corpus MLM training."""
from __future__ import annotations

import re

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast


SMILES_PATTERN = (
    r"\[[^\[\]]+\]|Br|Cl|B|C|N|O|P|S|F|I|b|c|n|o|s|p|"
    r"\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9]"
)
SPECIAL_TOKENS = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]


def split_smiles_atoms(smiles: str) -> list[str]:
    tokens = re.findall(SMILES_PATTERN, smiles)
    if "".join(tokens) != smiles:
        raise ValueError(f"SMILES contains unsupported token syntax: {smiles!r}")
    return tokens


def build_atom_tokenizer(train_smiles: list[str], max_length: int) -> PreTrainedTokenizerFast:
    """Fit only on training molecules; bracket atoms remain single tokens."""
    vocabulary = set()
    for smiles in train_smiles:
        vocabulary.update(split_smiles_atoms(smiles))
    tokens = SPECIAL_TOKENS + sorted(vocabulary)
    backend = Tokenizer(models.WordLevel(
        vocab={token: index for index, token in enumerate(tokens)}, unk_token="<unk>",
    ))
    backend.pre_tokenizer = pre_tokenizers.Split(Regex(SMILES_PATTERN), behavior="isolated")
    backend.decoder = decoders.Fuse()
    backend.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>", pair="<s> $A </s> </s> $B </s>",
        special_tokens=[("<s>", 0), ("</s>", 2)],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token="<s>", eos_token="</s>",
        cls_token="<s>", sep_token="</s>", pad_token="<pad>",
        unk_token="<unk>", mask_token="<mask>", model_max_length=max_length,
        model_input_names=["input_ids", "attention_mask"],
        clean_up_tokenization_spaces=False,
    )
