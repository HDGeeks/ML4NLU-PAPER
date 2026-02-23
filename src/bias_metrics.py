"""
bias_metrics.py

Intent
------
Geometric bias estimators for layer-wise analysis.

Provides:
- gender_direction(male, female): unit vector (male - female)
- projection_score(word_vec, direction): signed projection
- centroid_cosine_diff(word_vec, male_centroid, female_centroid): cos(m) - cos(f)
- spearman_rho(xs, ys): rank correlation (SciPy if available; pure-Python fallback)
"""

from __future__ import annotations

from typing import Sequence, Union, Optional
import math
import torch
import torch.nn.functional as F

TensorLike = Union[torch.Tensor, Sequence]


def _as_tensor(x: TensorLike) -> torch.Tensor:
    """
    Convert:
      - torch.Tensor -> tensor
      - list[Tensor] -> stack then mean
      - list[float]  -> tensor
    """
    if torch.is_tensor(x):
        return x

    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            raise ValueError("Empty list passed where a vector/centroid was expected.")

        # list of tensors -> mean stack
        if torch.is_tensor(x[0]):
            return torch.stack(list(x)).mean(0)

        # list of numbers -> tensor
        return torch.tensor(x, dtype=torch.float32)

    raise TypeError(f"Unsupported type for vector conversion: {type(x)}")


def _unit(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return v / (v.norm(p=2) + eps)


def gender_direction(male: TensorLike, female: TensorLike) -> torch.Tensor:
    """
    Compute a unit gender direction:
      g = unit( centroid(male) - centroid(female) )

    Accepts:
      - male, female as torch.Tensor centroids
      - OR as lists of vectors (list[Tensor])
      - OR as list[float] vectors (rare, but handled)
    """
    m = _as_tensor(male)
    f = _as_tensor(female)
    return _unit(m - f)


def projection_score(word_vec: torch.Tensor, direction: torch.Tensor) -> float:
    """
    Signed scalar projection of word_vec onto direction.
    Assumes direction is unit (we unit-normalize anyway).
    """
    d = _unit(direction)
    return float(torch.dot(word_vec, d))


def centroid_cosine_diff(
    word_vec: torch.Tensor,
    male_centroid: torch.Tensor,
    female_centroid: torch.Tensor,
) -> float:
    """
    Cosine difference estimator:
      cos(word, male_centroid) - cos(word, female_centroid)
    """
    w = _unit(word_vec)
    m = _unit(male_centroid)
    f = _unit(female_centroid)
    return float(torch.dot(w, m) - torch.dot(w, f))


def spearman_rho(xs: Sequence[float], ys: Sequence[float]) -> float:
    """
    Spearman rank correlation.
    Uses SciPy if installed; otherwise pure-Python fallback.
    """
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length.")
    n = len(xs)
    if n < 2:
        return float("nan")

    # Try SciPy first (best)
    try:
        from scipy.stats import spearmanr  # type: ignore
        return float(spearmanr(xs, ys).correlation)
    except Exception:
        pass

    # Fallback: compute ranks (average ranks for ties), then Pearson on ranks.
    def rankdata(a: Sequence[float]) -> list[float]:
        order = sorted(range(len(a)), key=lambda i: a[i])
        ranks = [0.0] * len(a)
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        return ranks

    rx = rankdata(xs)
    ry = rankdata(ys)

    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    denx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    deny = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    if denx == 0.0 or deny == 0.0:
        return float("nan")
    return float(num / (denx * deny))


if __name__ == "__main__":
    # Tiny sanity check if you run this file directly.
    w = torch.randn(5)
    m = torch.randn(5)
    f = torch.randn(5)
    g = gender_direction(m, f)
    print("projection_score:", projection_score(w, g))
    print("centroid_cosine_diff:", centroid_cosine_diff(w, m, f))