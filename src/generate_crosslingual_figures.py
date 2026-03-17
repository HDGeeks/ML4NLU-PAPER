"""
generate_crosslingual_figures.py

Produces two publication-quality figures from experiment CSVs:
  1. paper/images/crosslingual_peak_proj.png
     Grouped bar chart: peak projection per model × language
     Shows the mDeBERTa reversal holding cross-lingually.

  2. paper/images/crosslingual_mfn.png
     Stacked horizontal bar chart: M:F:N profession proportions
     at peak layer, per model × language.
"""

import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

BASE   = os.path.join(os.path.dirname(__file__), "..", "output")
OUTDIR = os.path.join(os.path.dirname(__file__), "..", "paper", "images")

MODELS = [
    ("XLM-R-base",   "xlm-roberta-base"),
    ("XLM-R-large",  "xlm-roberta-large"),
    ("XLM-V-base",   "facebook_xlm-v-base"),
    ("mDeBERTa",     "microsoft_mdeberta-v3-base"),
]
LANGS        = ["ti", "ar", "es"]
LANG_LABELS  = {"ti": "Tigrigna", "ar": "Arabic", "es": "Spanish"}

# layer-12 collapse: exclude from peak search
COLLAPSE_L12 = {
    ("ti", "facebook_xlm-v-base"),
    ("ti", "microsoft_mdeberta-v3-base"),
    ("ar", "facebook_xlm-v-base"),
    ("ar", "microsoft_mdeberta-v3-base"),
    ("es", "facebook_xlm-v-base"),
    ("es", "microsoft_mdeberta-v3-base"),
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def peak_proj(lang, slug):
    """Return (peak_layer, peak_mean_proj) excluding L12 where applicable."""
    path = os.path.join(BASE, lang, slug, f"{lang}_projection_layer_mean.csv")
    if not os.path.exists(path):
        return None, None
    rows = read_csv(path)
    excl = {12} if (lang, slug) in COLLAPSE_L12 else set()
    valid = [(int(r["layer"]), float(r["mean_proj"]))
             for r in rows if int(r["layer"]) not in excl]
    # for mDeBERTa pick the most extreme (largest |value|)
    if "mdeberta" in slug:
        best = max(valid, key=lambda x: abs(x[1]))
    else:
        best = max(valid, key=lambda x: x[1])
    return best


def mfn_mean_across_layers(lang, slug):
    """Return (n_male, n_female, n_neutral) using each profession's mean
    projection across all valid layers (excluding L12 collapse where applicable).
    More robust than a single peak layer."""
    path = os.path.join(BASE, lang, slug, f"{lang}_bias_by_layer.csv")
    if not os.path.exists(path):
        return None
    rows = read_csv(path)
    excl = {12} if (lang, slug) in COLLAPSE_L12 else set()
    # accumulate per-term across valid layers
    term_vals = {}
    for r in rows:
        if int(r["layer"]) in excl:
            continue
        t = r["term"]
        term_vals.setdefault(t, []).append(float(r["proj"]))
    # classify by sign of mean
    term_means = {t: sum(vs) / len(vs) for t, vs in term_vals.items()}
    # deduplicate (arquitecto duplicate in es)
    m = sum(1 for v in term_means.values() if v >  0.0)
    f = sum(1 for v in term_means.values() if v <  0.0)
    n = sum(1 for v in term_means.values() if v == 0.0)
    return m, f, n


# ── Collect data ──────────────────────────────────────────────────────────────

peak_data   = {}   # (label, lang) -> float
mfn_data    = {}   # (label, lang) -> (m, f, n)

for label, slug in MODELS:
    for lang in LANGS:
        pl, pv = peak_proj(lang, slug)
        peak_data[(label, lang)] = pv   # None if model not run for that lang
        mfn = mfn_mean_across_layers(lang, slug)
        if mfn is not None:
            mfn_data[(label, lang)] = mfn


# ── Figure 1: Grouped bar — Peak Projection ───────────────────────────────────

LANG_COLORS = {
    "ti": "#2E86AB",   # steel blue  — primary (Tigrigna)
    "ar": "#E07A5F",   # terracotta  — Arabic
    "es": "#3D405B",   # dark slate  — Spanish
}

fig, ax = plt.subplots(figsize=(9, 5))

n_models = len(MODELS)
n_langs  = len(LANGS)
group_w  = 0.7
bar_w    = group_w / n_langs
x        = np.arange(n_models)

for i, lang in enumerate(LANGS):
    vals    = [peak_data.get((lbl, lang)) for lbl, _ in MODELS]
    offsets = (i - n_langs / 2 + 0.5) * bar_w
    bars    = ax.bar(
        x + offsets, vals,
        width=bar_w * 0.92,
        color=LANG_COLORS[lang],
        label=LANG_LABELS[lang],
        zorder=3,
    )
    # value labels
    for bar, v in zip(bars, vals):
        if v is None:
            continue
        va  = "bottom" if v >= 0 else "top"
        pad = 0.05 if v >= 0 else -0.05
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + pad,
            f"{v:+.2f}",
            ha="center", va=va,
            fontsize=7.5, color="#222222",
        )

ax.axhline(0, color="black", linewidth=0.8, zorder=2)
ax.set_xticks(x)
ax.set_xticklabels([lbl for lbl, _ in MODELS], fontsize=11)
ax.set_ylabel("Peak Mean Projection Score", fontsize=11)
ax.set_title(
    "Peak Gender Bias by Architecture and Language\n"
    r"(positive $=$ male-leaning, negative $=$ female-leaning)",
    fontsize=12, pad=10,
)
ax.legend(loc="upper right", fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# shade mDeBERTa column to highlight reversal
ax.axvspan(2.5, 3.5, color="#f5f0ff", zorder=0)
ax.text(3, ax.get_ylim()[0] * 0.85, "reversal",
        ha="center", fontsize=8, color="#7b5ea7", style="italic")

plt.tight_layout()
out1 = os.path.join(OUTDIR, "crosslingual_peak_proj.png")
plt.savefig(out1, dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved: {out1}")


# ── Figure 2: Stacked horizontal bar — M:F:N proportions ─────────────────────

# Layout: one row per model, one panel per language (3 columns)
fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)

M_COLOR = "#2E86AB"
F_COLOR = "#E07A5F"
N_COLOR = "#CCCCCC"

y_labels = [lbl for lbl, _ in MODELS]
y_pos    = np.arange(n_models)

for col, lang in enumerate(LANGS):
    ax = axes[col]
    m_vals, f_vals, n_vals, totals = [], [], [], []

    for label, slug in MODELS:
        entry = mfn_data.get((label, lang))
        if entry is None:
            m_vals.append(0); f_vals.append(0); n_vals.append(0); totals.append(0)
        else:
            m, f, n = entry
            tot = m + f + n
            m_vals.append(m / tot * 100 if tot else 0)
            f_vals.append(f / tot * 100 if tot else 0)
            n_vals.append(n / tot * 100 if tot else 0)
            totals.append(tot)

    bar_h = 0.55
    bm = ax.barh(y_pos, m_vals, height=bar_h, color=M_COLOR, label="Male-leaning")
    bf = ax.barh(y_pos, f_vals, height=bar_h, left=m_vals, color=F_COLOR, label="Female-leaning")
    bn = ax.barh(y_pos, n_vals, height=bar_h,
                 left=[a + b for a, b in zip(m_vals, f_vals)],
                 color=N_COLOR, label="Neutral")

    # percentage annotations inside bars
    for i, (m, f, n, tot) in enumerate(zip(m_vals, f_vals, n_vals, totals)):
        if tot == 0:
            continue
        if m > 8:
            ax.text(m / 2, i, f"{m:.0f}%", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
        if f > 8:
            ax.text(m + f / 2, i, f"{f:.0f}%", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")

    ax.set_xlim(0, 100)
    ax.set_title(LANG_LABELS[lang], fontsize=12, pad=6)
    ax.set_xlabel("% of professions", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    if col == 0:
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=10.5)

legend_patches = [
    mpatches.Patch(color=M_COLOR, label="Male-leaning"),
    mpatches.Patch(color=F_COLOR, label="Female-leaning"),
    mpatches.Patch(color=N_COLOR, label="Neutral"),
]
fig.legend(
    handles=legend_patches,
    loc="lower center", ncol=3,
    fontsize=10, frameon=False,
    bbox_to_anchor=(0.5, -0.04),
)
fig.suptitle(
    "Profession Bias Direction at Peak Layer: M:F:N Proportions",
    fontsize=13, y=1.01,
)

plt.tight_layout()
out2 = os.path.join(OUTDIR, "crosslingual_mfn.png")
plt.savefig(out2, dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")
