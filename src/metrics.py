"""
metrics.py
----------
Bias estimators and Spearman agreement.

All functions are pure-numeric: no I/O, no model loading.
Inputs are tensors (from embeddings.py) and lists; outputs are floats/dicts.
"""

from collections import defaultdict

import torch
from scipy.stats import spearmanr

from data_loader import find_sentences
from embeddings import word_vector_per_layer


def projection_score(vec, direction) -> float:
    """
    Scalar projection of a profession vector onto the unit gender direction.
    ProjBias(p) = dot(s(p), g / ||g||)
    Positive = male-leaning; negative = female-leaning.
    """
    return float(torch.dot(vec, direction))


def centroid_cosine_diff(vec, male_centroid, female_centroid) -> float:
    """
    CosBias(p) = cosine(s(p), c_M) - cosine(s(p), c_F)
    Magnitude-normalised complement to projection_score (used for RQ3).
    """
    def _cos(a, b):
        return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-8))
    return _cos(vec, male_centroid) - _cos(vec, female_centroid)


def bias_scores_for_word(model, tokenizer, corpus: list, word: str,
                          male_centroids, female_centroids, directions,
                          n_contexts: int, verbose: bool = False):
    """
    Compute per-layer bias scores for a profession term.

    Returns
    -------
    (scores, n_found)
        scores   : list of {layer, proj, cosdiff} dicts, or None if skipped
        n_found  : number of corpus sentences found for this word
    """
    if verbose:
        print("\n" + "-"*60)
        print(f"STEP: Scoring '{word}'")

    sentences = find_sentences(corpus, word, n_contexts, tokenizer, verbose=verbose)
    if len(sentences) < n_contexts:
        return None, len(sentences)

    layer_vecs = word_vector_per_layer(model, tokenizer, sentences, word, verbose=verbose)
    if layer_vecs is None:
        return None, 0

    scores = []
    if verbose:
        print(f"\n  {'Layer':>6}  {'proj':>10}  {'cosdiff':>10}  interpretation")
        print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*30}")

    for i, vec in enumerate(layer_vecs):
        proj    = projection_score(vec, directions[i])
        cosdiff = centroid_cosine_diff(vec, male_centroids[i], female_centroids[i])
        if verbose:
            interp = ("approx neutral" if abs(proj) < 0.05 else
                      f"male-leaning  (+{proj:.3f})" if proj > 0 else
                      f"female-leaning ({proj:.3f})")
            print(f"  {i:>6}  {proj:>10.4f}  {cosdiff:>10.4f}  {interp}")
        scores.append({"layer": i, "proj": proj, "cosdiff": cosdiff})

    if verbose:
        projs = [s["proj"] for s in scores]
        print(f"\n  Mean proj across all layers: {sum(projs)/len(projs):.4f}")

    return scores, len(sentences)


def spearman_per_layer(all_records: list) -> list:
    """
    Spearman rank correlation between proj and cosdiff at each layer.
    Returns a list of {layer, rho, p, n} dicts sorted by layer.
    """
    by_layer = defaultdict(list)
    for r in all_records:
        by_layer[r["layer"]].append(r)

    rows = []
    for layer, records in sorted(by_layer.items()):
        projs    = [r["proj"]    for r in records]
        cosdiffs = [r["cosdiff"] for r in records]
        if len(projs) < 2:
            continue
        rho, p = spearmanr(projs, cosdiffs)
        rows.append({"layer": layer, "rho": rho, "p": p, "n": len(projs)})
    return rows
