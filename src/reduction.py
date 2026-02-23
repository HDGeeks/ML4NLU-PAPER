"""
reduction.py

Intent
------
Reduce contextual transformer representations to "static-like" embeddings.

Given:
- a target word (e.g., "médico")
- N contexts (sentences) containing that word
- a transformer model that can return hidden states

We produce:
- one vector per layer (including embeddings layer 0)
- by mean-pooling over subword pieces for the target word
- then mean-aggregating across contexts

This is the core building block for:
- layer-wise bias measurement
- applying classic static-embedding bias metrics to contextual models
"""

from __future__ import annotations

from typing import List
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def layerwise_static_embedding(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    contexts: List[str],
    target: str,
) -> List[torch.Tensor]:
    """
    Return a list of tensors: [layer0_vec, layer1_vec, ..., layerL_vec]
    Each vector is (hidden_size,) and represents `target` averaged over contexts.
    """
    model.eval()
    per_layer: List[List[torch.Tensor]] | None = None

    with torch.no_grad():
        for text in contexts:
            # Find the character span of the target in the raw string
            start = text.find(target)
            if start < 0:
                continue
            end = start + len(target)

            enc = tokenizer(
                text,
                return_tensors="pt",
                return_offsets_mapping=True,
                add_special_tokens=True,
            )

            offsets = enc.pop("offset_mapping")[0].tolist()

            # Token indices whose character offsets overlap [start, end)
            idxs = [
                i for i, (a, b) in enumerate(offsets)
                if (a != b) and not (b <= start or a >= end)
            ]
            if not idxs:
                continue

            out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states  # tuple: (layer0, layer1, ..., layerL)

            if per_layer is None:
                per_layer = [[] for _ in range(len(hs))]

            for l, layer in enumerate(hs):
                # layer: (1, seq_len, hidden_size)
                token_vecs = layer[0, idxs, :]          # (k, hidden)
                per_layer[l].append(token_vecs.mean(0)) # (hidden,)

    if not per_layer or any(len(v) == 0 for v in per_layer):
        raise ValueError("No valid target spans found in the provided contexts.")

    # Mean across contexts -> one vector per layer
    return [torch.stack(v).mean(0) for v in per_layer]