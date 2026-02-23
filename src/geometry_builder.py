"""
geometry_builder.py

Intent
------
Build layer-wise gender geometry (male/female centroids and gender directions)
from anchor terms, using the same contextual-to-static reduction as profession terms.
"""

from __future__ import annotations

from typing import List, Tuple
import torch

from reduction import layerwise_static_embedding
from bias_metrics import gender_direction


def _find_contexts(corpus: list[str], term: str, n: int) -> list[str]:
    return [s for s in corpus if term in s][:n]


@torch.no_grad()
def build_gender_geometry_per_layer(
    model,
    tok,
    corpus: list[str],
    male_terms: list[str],
    female_terms: list[str],
    n_anchor_contexts: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Returns:
      male_centroids   : list[Tensor] length L
      female_centroids : list[Tensor] length L
      directions       : list[Tensor] length L, each unit(m - f)
    """

    male_layer_vecs: List[List[torch.Tensor]] | None = None
    female_layer_vecs: List[List[torch.Tensor]] | None = None

    # Collect male anchors
    for t in male_terms:
        ctx = _find_contexts(corpus, t, n_anchor_contexts)
        if not ctx:
            continue
        layerwise = layerwise_static_embedding(model, tok, ctx, t)  # list[Tensor] length L
        if male_layer_vecs is None:
            male_layer_vecs = [[] for _ in range(len(layerwise))]
        for ell, v in enumerate(layerwise):
            male_layer_vecs[ell].append(v)

    # Collect female anchors
    for t in female_terms:
        ctx = _find_contexts(corpus, t, n_anchor_contexts)
        if not ctx:
            continue
        layerwise = layerwise_static_embedding(model, tok, ctx, t)
        if female_layer_vecs is None:
            female_layer_vecs = [[] for _ in range(len(layerwise))]
        for ell, v in enumerate(layerwise):
            female_layer_vecs[ell].append(v)

    if male_layer_vecs is None or female_layer_vecs is None:
        raise ValueError("No anchor contexts found. Check anchor lists and corpus contents.")

    # Compute centroids per layer
    male_centroids: List[torch.Tensor] = []
    female_centroids: List[torch.Tensor] = []

    L = len(male_layer_vecs)
    if len(female_layer_vecs) != L:
        raise ValueError("Layer count mismatch between male and female anchor reductions.")

    for ell in range(L):
        if len(male_layer_vecs[ell]) == 0 or len(female_layer_vecs[ell]) == 0:
            raise ValueError(f"Empty anchor vectors at layer {ell}. Increase n_anchor_contexts or fix corpus.")
        male_centroids.append(torch.stack(male_layer_vecs[ell]).mean(0))
        female_centroids.append(torch.stack(female_layer_vecs[ell]).mean(0))

    # Compute directions per layer (NOW tensors, not lists)
    directions = [gender_direction(male_centroids[ell], female_centroids[ell]) for ell in range(L)]

    return male_centroids, female_centroids, directions