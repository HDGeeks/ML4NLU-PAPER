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

    print("\n===== DEBUG START =====")
    print("Target:", target)
    print("Target repr:", repr(target))
    print("Number of contexts:", len(contexts))
    print("=======================\n")

    model.eval()
    per_layer: List[List[torch.Tensor]] | None = None
    successful_contexts = 0

    with torch.no_grad():
        for idx_ctx, text in enumerate(contexts):

            print(f"\n--- Context {idx_ctx} ---")
            print("Text:", text)
            print("Text repr:", repr(text))

            # 1. Raw substring search
            start = text.find(target)
            print("find() result:", start)

            if start < 0:
                print("Skipping: target not found in raw string.")
                continue

            end = start + len(target)
            print("Char span:", (start, end))

            # 2. Tokenize
            enc = tokenizer(
                text,
                return_tensors="pt",
                return_offsets_mapping=True,
                add_special_tokens=True,
            )

            offsets = enc.pop("offset_mapping")[0].tolist()
            print("Number of tokens:", len(offsets))

            # 3. Overlapping token indices
            idxs = [
                i for i, (a, b) in enumerate(offsets)
                if (a != b) and not (b <= start or a >= end)
            ]

            print("Matched token indices:", idxs)

            if not idxs:
                print("Skipping: no overlapping token span.")
                continue

            # 4. Forward pass
            out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states

            if per_layer is None:
                per_layer = [[] for _ in range(len(hs))]

            for l, layer in enumerate(hs):
                token_vecs = layer[0, idxs, :]
                per_layer[l].append(token_vecs.mean(0))

            successful_contexts += 1
            print("Context processed successfully.")

    print("\n===== SUMMARY =====")
    print("Successful contexts:", successful_contexts)
    print("===================\n")

    if not per_layer or any(len(v) == 0 for v in per_layer):
        print("Failure: no valid spans collected.")
        raise ValueError("No valid target spans found in the provided contexts.")

    return [torch.stack(v).mean(0) for v in per_layer]