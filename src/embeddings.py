"""
embeddings.py
-------------
Model forward passes: extract per-layer word vectors and build
the layer-wise gender geometry (centroids + unit direction).
"""

import torch
from data_loader import find_sentences


def word_vector_per_layer(model, tokenizer, sentences: list, word: str,
                           verbose: bool = False):
    """
    For each sentence, locate the target word's subword tokens, mean-pool them,
    then average across all sentences. Returns one tensor per transformer layer,
    or None if the word could not be located in any sentence.
    """
    word_tokens = tokenizer.tokenize(word)
    if verbose:
        print(f"\n  word_vector_per_layer('{word}')")
        print(f"    Processing {len(sentences)} sentences across all layers...")

    accumulators = None
    count = 0

    for sent_idx, sentence in enumerate(sentences):
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        sent_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        pos = None
        for i in range(len(sent_tokens) - len(word_tokens) + 1):
            if sent_tokens[i : i + len(word_tokens)] == word_tokens:
                pos = (i, i + len(word_tokens))
                break

        if pos is None:
            if verbose:
                print(f"    [sent {sent_idx+1}] word not found in token list — skipping")
            continue

        start, end = pos
        if verbose:
            print(f"    [sent {sent_idx+1}] tokens [{start}:{end}]: {sent_tokens[start:end]}")

        layer_vecs = [h[0, start:end].mean(dim=0) for h in outputs.hidden_states]

        if accumulators is None:
            accumulators = layer_vecs
        else:
            accumulators = [a + b for a, b in zip(accumulators, layer_vecs)]
        count += 1

    if count == 0 or accumulators is None:
        if verbose:
            print(f"    No valid sentences for '{word}'")
        return None

    averaged = [v / count for v in accumulators]
    if verbose:
        print(f"    Averaged {count} sentences -> {len(averaged)} layer vectors")
    return averaged


def build_gender_geometry(model, tokenizer, corpus: list,
                           male_words: list, female_words: list,
                           n_contexts: int, verbose: bool = False):
    """
    Build male centroid, female centroid, and unit gender direction per layer.
    Returns (male_centroids, female_centroids, directions).
    """
    if verbose:
        print("\n" + "-"*60)
        print("STEP: Building gender geometry")
        print(f"  Male anchors   ({len(male_words)}): {male_words}")
        print(f"  Female anchors ({len(female_words)}): {female_words}")
        print(f"  Contexts per anchor: {n_contexts}")

    def _centroid(words, label):
        total   = None
        count   = 0
        skipped = []
        for word in words:
            sents = find_sentences(corpus, word, n_contexts, tokenizer, verbose=verbose)
            if len(sents) < n_contexts:
                skipped.append((word, len(sents)))
                continue
            vecs = word_vector_per_layer(model, tokenizer, sents, word, verbose=verbose)
            if vecs is None:
                skipped.append((word, 0))
                continue
            total = vecs if total is None else [a + b for a, b in zip(total, vecs)]
            count += 1
        if verbose:
            print(f"\n  {label} centroid: built from {count}/{len(words)} anchors")
            if skipped:
                print(f"  Skipped: {skipped}")
        if count == 0:
            raise ValueError(f"No {label} anchor words had enough corpus sentences.")
        return [v / count for v in total]

    male_centroids   = _centroid(male_words,   "Male")
    female_centroids = _centroid(female_words, "Female")

    directions = []
    for i, (m, f) in enumerate(zip(male_centroids, female_centroids)):
        diff = m - f
        norm = diff.norm()
        if norm < 1e-8:
            raise ValueError(f"Male and female centroids are identical at layer {i}.")
        directions.append(diff / norm)

    if verbose:
        print(f"\n  Gender direction built for {len(directions)} layers")
        sample = [0, 4, 8, 12] if len(directions) > 12 else list(range(len(directions)))
        print("  Centroid separation per sample layer:")
        for l in sample:
            print(f"    Layer {l:2d}: {(male_centroids[l] - female_centroids[l]).norm():.4f}")

    return male_centroids, female_centroids, directions
