"""
analysis.py

Intent
------
Small analysis helpers (RQ3):
- Spearman agreement between estimators
- Simple aggregations for Results tables
"""

from __future__ import annotations
from typing import Iterable, Tuple, Dict, List
import numpy as np

try:
    from scipy.stats import spearmanr
except ImportError as e:
    raise ImportError(
        "Missing dependency: scipy.\n"
        "Install: pip install scipy"
    ) from e


def spearman_agreement(x: Iterable[float], y: Iterable[float]) -> Tuple[float, float]:
    """
    Spearman rank correlation between two score lists.
    Returns (rho, p_value).

    Notes:
    - If either vector is constant (no variance), Spearman is undefined.
      We return (nan, nan) instead of emitting ConstantInputWarning.
    """
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)

    # Not enough data
    if x.size < 2 or y.size < 2:
        return float("nan"), float("nan")

    # Constant input => undefined correlation
    # (covers cases like all zeros, or all identical floats)
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan"), float("nan")

    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def layerwise_spearman(records: List[dict]) -> List[dict]:
    """
    records: list of dict rows containing at least:
      - 'layer' (int)
      - 'proj' (float)
      - 'cosdiff' (float)

    Returns rows:
      - layer, spearman_rho, spearman_p, n
    """
    by_layer: Dict[int, List[dict]] = {}
    for r in records:
        layer = int(r["layer"])
        by_layer.setdefault(layer, []).append(r)

    out: List[dict] = []
    for layer in sorted(by_layer.keys()):
        rows = by_layer[layer]

        # be robust if values come in as strings
        proj = [float(rr["proj"]) for rr in rows]
        cosd = [float(rr["cosdiff"]) for rr in rows]

        rho, p = spearman_agreement(proj, cosd)
        out.append(
            {
                "layer": layer,
                "spearman_rho": rho,
                "spearman_p": p,
                "n": len(rows),
            }
        )

    return out