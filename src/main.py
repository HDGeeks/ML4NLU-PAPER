"""
main.py

Intent
------
CPU-friendly multilingual bias pipeline (paper-aligned, minimal main).

What it does
------------
1) Load corpus + inventories (professions + gender anchors)
2) Load pretrained multilingual encoder (mBERT / XLM-R)
3) Build *layer-wise* gender geometry from anchors:
     - male centroid cM_l
     - female centroid cF_l
     - gender direction g_l
4) For each profession term:
     - contextual-to-static reduction per layer (Bommasani et al. style)
     - compute two bias estimators per layer:
         (a) projection onto g_l
         (b) centroid cosine difference vs (cM_l, cF_l)
5) Save:
     - bias scores CSV (per profession x layer)
     - Spearman agreement CSV (per layer, across professions)
6) Plot:
     - layer-wise mean projection curve (PNG)
     - layer-wise mean projection CSV

Design rules
------------
- No fine-tuning
- CPU-only
- Deterministic (fixed slicing of contexts from corpus)
- All knobs live in the CONFIG BLOCK

Requires
--------
- scipy (for Spearman): pip install scipy
"""

from __future__ import annotations

import os
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
from transformers import AutoTokenizer, AutoModel

from data_loader import load_corpus, load_anchors, load_professions
from reduction import layerwise_static_embedding
from bias_metrics import projection_score, centroid_cosine_diff
from geometry_builder import build_gender_geometry_per_layer
from analysis import layerwise_spearman
from plot_layer_projection import plot_projection_curve


# ============================================================
# ====================== CONFIG BLOCK ========================
# ============================================================

@dataclass(frozen=True)
class Config:
    LANG: str = "en"  # "es" | "ar" | "ti"
    MODEL_NAME: str = "bert-base-multilingual-cased"
    TARGET_WORD: str | None = None
    N_TARGET_CONTEXTS: int = 3
    N_ANCHOR_CONTEXTS: int = 3

    OUT_DIR: str = "output"   # base output dir

    @property
    def out_lang_dir(self) -> Path:
        return Path(self.OUT_DIR) / self.LANG

    @property
    def out_bias_csv(self) -> Path:
        return self.out_lang_dir / f"{self.LANG}_{self.MODEL_NAME}_bias_by_layer.csv"

    @property
    def out_spearman_csv(self) -> Path:
        return self.out_lang_dir / f"{self.LANG}_{self.MODEL_NAME}_spearman_by_layer.csv"

    @property
    def out_fig_png(self) -> Path:
        return self.out_lang_dir / "figs" / f"{self.LANG}_{self.MODEL_NAME}_projection_curve.png"

    @property
    def out_layer_mean_csv(self) -> Path:
        return self.out_lang_dir / f"{self.LANG}_{self.MODEL_NAME}_projection_layer_mean.csv"

    @property
    def plot_title(self) -> str:
        return f"Layer-wise mean projection ({self.LANG} | {self.MODEL_NAME})"


CFG = Config()

# ============================================================


# Keep output clean
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"


def find_contexts(corpus: list[str], term: str, n: int, tokenizer) -> list[str]:
    matches = []

    term_tokens = tokenizer.tokenize(term)

    for sentence in corpus:
        sent_tokens = tokenizer.tokenize(sentence)

        # sliding window token match
        for i in range(len(sent_tokens) - len(term_tokens) + 1):
            if sent_tokens[i:i + len(term_tokens)] == term_tokens:
                matches.append(sentence)
                break

        if len(matches) >= n:
            break

    return matches


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _bias_rows_for_term(
    model,
    tok,
    corpus: List[str],
    term: str,
    n_contexts: int,
    male_centroids: List[torch.Tensor],
    female_centroids: List[torch.Tensor],
    directions: List[torch.Tensor],
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Returns:
      rows: list of dicts with layer-wise bias scores
      n_found: number of contexts found for this term
    """
    ctx = find_contexts(corpus, term, n_contexts, tok)
    if len(ctx) < n_contexts:
        return [], len(ctx)

    layer_vecs = layerwise_static_embedding(model, tok, ctx, term)

    rows: List[Dict[str, Any]] = []
    for ell, vec in enumerate(layer_vecs):
        proj = projection_score(vec, directions[ell])
        cosd = centroid_cosine_diff(vec, male_centroids[ell], female_centroids[ell])
        rows.append({
            "layer": ell,
            "proj": proj,
            "cosdiff": cosd,
        })
    return rows, len(ctx)


def main() -> None:
    # Ensure output dirs exist
    _ensure_parent_dir(CFG.out_bias_csv)
    _ensure_parent_dir(CFG.out_spearman_csv)
    _ensure_parent_dir(CFG.out_fig_png)
    _ensure_parent_dir(CFG.out_layer_mean_csv)

    # -----------------------------
    # Load data
    # -----------------------------
    corpus = load_corpus(CFG.LANG)
    male_terms, female_terms = load_anchors(CFG.LANG)
    professions = load_professions(CFG.LANG)

    # -----------------------------
    # Load model
    # -----------------------------
    tok = AutoTokenizer.from_pretrained(CFG.MODEL_NAME, trust_remote_code=False)
    model = AutoModel.from_pretrained(CFG.MODEL_NAME, trust_remote_code=False)
    model.eval()

    # -----------------------------
    # Build gender geometry per layer (paper-aligned)
    # -----------------------------
    with torch.no_grad():
        male_centroids, female_centroids, directions = build_gender_geometry_per_layer(
            model=model,
            tok=tok,
            corpus=corpus,
            male_terms=male_terms,
            female_terms=female_terms,
            n_anchor_contexts=CFG.N_ANCHOR_CONTEXTS,
        )

    # -----------------------------
    # Optional debug: single word curve
    # -----------------------------
    if CFG.TARGET_WORD:
        rows, found = _bias_rows_for_term(
            model=model,
            tok=tok,
            corpus=corpus,
            term=CFG.TARGET_WORD,
            n_contexts=CFG.N_TARGET_CONTEXTS,
            male_centroids=male_centroids,
            female_centroids=female_centroids,
            directions=directions,
        )
        if not rows:
            raise ValueError(
                f"Need ≥{CFG.N_TARGET_CONTEXTS} contexts for '{CFG.TARGET_WORD}', found {found}"
            )

        print(f"\nLayer-wise bias scores for: {CFG.TARGET_WORD}")
        for r in rows:
            print(f"Layer {r['layer']:02d}: proj={r['proj']:+.4f}  cosdiff={r['cosdiff']:+.4f}")

    # -----------------------------
    # Full profession export
    # -----------------------------
    all_bias_records: List[Dict[str, Any]] = []
    skipped: List[Tuple[str, int]] = []
    kept = 0

    for prof in professions:
        rows, found = _bias_rows_for_term(
            model=model,
            tok=tok,
            corpus=corpus,
            term=prof,
            n_contexts=CFG.N_TARGET_CONTEXTS,
            male_centroids=male_centroids,
            female_centroids=female_centroids,
            directions=directions,
        )
        if not rows:
            skipped.append((prof, found))
            continue

        for r in rows:
            all_bias_records.append({
                "language": CFG.LANG,
                "model": CFG.MODEL_NAME,
                "term": prof,
                "layer": r["layer"],
                "proj": r["proj"],
                "cosdiff": r["cosdiff"],
            })
        kept += 1

    # -----------------------------
    # Write bias CSV
    # -----------------------------
    with CFG.out_bias_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["language", "model", "term", "layer", "proj", "cosdiff"])
        for r in all_bias_records:
            w.writerow([r["language"], r["model"], r["term"], r["layer"], r["proj"], r["cosdiff"]])

    # -----------------------------
    # Spearman agreement per layer (RQ3)
    # Spearman is computed across professions for each layer.
    # -----------------------------
    spearman_input = [
        {"layer": r["layer"], "proj": r["proj"], "cosdiff": r["cosdiff"]}
        for r in all_bias_records
    ]
    spearman_rows = layerwise_spearman(spearman_input)

    with CFG.out_spearman_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["language", "model", "layer", "spearman_rho", "spearman_p", "n"])
        for row in spearman_rows:
            w.writerow([CFG.LANG, CFG.MODEL_NAME, row["layer"], row["spearman_rho"], row["spearman_p"], row["n"]])

    # -----------------------------
    # Plot layer-wise mean projection curve
    # (from the bias CSV, using column "proj")
    # -----------------------------
    plot_projection_curve(
        in_csv=str(CFG.out_bias_csv),
        out_png=str(CFG.out_fig_png),
        out_mean_csv=str(CFG.out_layer_mean_csv),
        title=CFG.plot_title,
        value_col="proj",
    )

    print(f"\nSaved: {CFG.out_bias_csv}")
    print(f"Saved: {CFG.out_spearman_csv}")
    print(f"Saved plot: {CFG.out_fig_png}")
    print(f"Saved layer-mean CSV: {CFG.out_layer_mean_csv}")
    print(f"Professions kept: {kept}/{len(professions)}")

    if skipped:
        print("Skipped (profession -> contexts found):")
        for prof, c in skipped:
            print(f"  - {prof}: {c}")


if __name__ == "__main__":
    main()