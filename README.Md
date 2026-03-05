# Layer-Wise Gender Bias in Multilingual Foundation Models
### A Five-Architecture Cross-Lingual Analysis Pipeline

> **Paper:** *Layer-Wise Gender Bias in Multilingual Foundation Models: A Five-Architecture Cross-Lingual Analysis*
> **Author:** Daniel Tesfai Kebede (Matr. 1716694)
> **Course:** Trends in Machine Learning — University of Trier
> **Languages analysed:** Spanish · Arabic · Tigrigna (primary)
> **Models evaluated:** 5 multilingual transformer architectures
> **Hardware requirement:** CPU only — no GPU, no fine-tuning

---

## Table of Contents

1. [Research Overview](#1-research-overview)
2. [Repository Layout](#2-repository-layout)
3. [Data Layout](#3-data-layout)
4. [Pipeline — Step by Step](#4-pipeline--step-by-step)
5. [Source Code Map](#5-source-code-map)
6. [Configuration Reference](#6-configuration-reference)
7. [How to Run](#7-how-to-run)
8. [Output Files Explained](#8-output-files-explained)
9. [Five-Model Results Summary](#9-five-model-results-summary)
10. [Run Modes](#10-run-modes)
11. [Design Decisions](#11-design-decisions)
12. [Dependencies](#12-dependencies)

---

## 1. Research Overview

Multilingual transformer models embed dozens of languages into a shared
representational space. This raises a structural question: **do social associations
(e.g. gender stereotypes encoded in high-resource languages) transfer into
low-resource languages that barely appear in training data?**

This project answers that question for **Tigrigna** — a Ge'ez-script Semitic
language spoken in Eritrea and northern Ethiopia — which has essentially zero
dedicated presence in any multilingual pretraining corpus.

### Research Questions

| # | Question |
|---|----------|
| RQ1 | Do multilingual transformers encode measurable gender bias for Arabic, Spanish, and Tigrigna under a unified geometric protocol? |
| RQ2 | How does bias vary across transformer layers, and at which depths do associations peak? |
| RQ3 | Do two independent geometric estimators agree on bias trends across languages and models? |
| RQ4 | Does attention architecture determine the *direction* of gender bias, independent of pretraining data? |

### Key Findings

- **4 of 5 models** encode male-leaning gender bias for Tigrigna professions, despite
  the language being absent from pretraining data → cross-lingual transfer of bias confirmed.
- Bias **builds progressively** across layers, peaking at mid-to-upper layers (layer 9–11
  for base-size models, layer 23 for XLM-R-large), then collapses at the final layer.
- Two independent estimators agree at **Spearman ρ ≥ 0.970** across all valid models,
  confirming methodological robustness.
- **mDeBERTa-v3-base** produces a complete directional reversal: 28 of 30 profession terms
  are *female-leaning*. This is attributed to its Disentangled Attention mechanism —
  the only systematic architectural difference from the XLM-R family.

---

## 2. Repository Layout

```
ML4NLU-PAPER/
│
├── src/                         ← All Python source code
│   ├── config.py                ← Single settings file — edit this to run experiments
│   ├── data_loader.py           ← Load corpus / anchors / professions from disk
│   ├── embeddings.py            ← Model forward passes → layer-wise word vectors
│   ├── metrics.py               ← Bias estimators + Spearman agreement
│   ├── output.py                ← Tee logger + projection curve PNG/CSV
│   ├── debug_mode.py            ← Verbose single-word trace mode
│   ├── main.py                  ← Orchestration entry point
│   │
│   ├── check_rep.py             ← Standalone: representation uniqueness analysis
│   ├── target_weights.py        ← Standalone: static weight-level word inspection
│   ├── diagnsotic.py            ← Standalone: tokenizer coverage of professions
│   │
│   ├── tigrigna_corpus_builder.py     ← Generated the Tigrigna corpus (v1)
│   ├── tigrigna_corpus_builder_v2.py  ← Generated the Tigrigna corpus (v2, used)
│   ├── tigrigna_anchor_builder.py     ← Generated the anchor word CSV
│   └── prepare_tigrigna_corpus.py     ← Pre-processed raw corpus text
│
├── data/
│   ├── tigrigna/
│   │   ├── corpus_ti.txt              ← Tigrigna corpus (one sentence per line)
│   │   ├── professions_ti.csv         ← 30 profession terms in Tigrigna
│   │   └── anchors_ti.csv             ← 11 male + 10 female anchor words
│   ├── arabic/
│   │   ├── corpus_ar.txt
│   │   ├── professions_ar.csv
│   │   └── anchors_ar.csv
│   └── spanish/
│       ├── corpus_es.txt
│       ├── professions_es.csv
│       └── anchors_es.csv
│
├── output/
│   └── ti/
│       ├── xlm-roberta-base/
│       │   ├── ti_bias_by_layer.csv
│       │   ├── ti_spearman_by_layer.csv
│       │   ├── ti_projection_layer_mean.csv
│       │   └── figs/ti_xlmr_base_projection_curve.png
│       ├── xlm-roberta-large/        ← same structure
│       ├── facebook_xlm-v-base/      ← same structure
│       └── microsoft_mdeberta-v3-base/ ← same structure
│
├── paper/
│   ├── main_v2.tex                   ← Full LaTeX paper (XeLaTeX)
│   ├── refs.bib                      ← Bibliography
│   └── images/                       ← All 5 projection curve figures (Overleaf-ready)
│
├── .models/hub/                      ← HuggingFace model checkpoints (local cache)
└── README.md                         ← This file
```

---

## 3. Data Layout

### Corpus (`corpus_{lang}.txt`)

Plain text, **one sentence per line**. The Tigrigna corpus was template-generated
to contain balanced male and female co-occurrence contexts for each profession.

```
ሰበይቲ ሓኪም እያ
ሓወይ ሓኪም እዩ
ንሳ ብቑዓት ሓኪም እያ
...
```

Total: ~14,850 sentences for Tigrigna.

### Professions (`professions_{lang}.csv`)

```
profession_id,profession_lang,profession
ti_001,ti,ሓኪም
ti_002,ti,ነርስ
ti_003,ti,መምህር
...
```

30 terms per language. The `profession` column is the target word.

### Anchors (`anchors_{lang}.csv`)

```
gender,type,subtype,term,english_ref
male,pronoun,subject,ንሱ,he
male,kinship,sibling,ሓወይ,my brother
female,pronoun,subject,ንሳ,she
female,kinship,sibling,ሓፍተይ,my sister
...
```

11 male + 10 female Tigrigna terms (pronouns, kinship terms, generic gender markers).
Validated by the author, a native Tigrigna speaker.

---

## 4. Pipeline — Step by Step

The full pipeline runs once per model. Every step feeds directly into the next.

```
┌─────────────────────────────────────────────────────────────────────┐
│  config.py                                                          │
│  ─────────────────────────────────────────────────────────────────  │
│  LANGUAGE = "ti"          MODE = "bulk"                             │
│  MODEL_NAMES = [          ANCHOR_CONTEXTS     = 12                  │
│    "xlm-roberta-base",    PROFESSION_CONTEXTS = 20                  │
│    "xlm-roberta-large",   DEBUG_WORD = "ሓረስታይ"                     │
│    ...                                                              │
│  ]                                                                  │
└───────────────────────────────┬─────────────────────────────────────┘
                                │  read by main.py at startup
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1 — Load Data                             [data_loader.py]    │
│  ─────────────────────────────────────────────────────────────────  │
│  load_corpus("ti")        → list of ~14,850 sentences              │
│  load_anchors("ti")       → male_words (11), female_words (10)     │
│  load_professions("ti")   → job_titles (30 terms)                  │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 2 — Load Model Checkpoint                 [main.py]           │
│  ─────────────────────────────────────────────────────────────────  │
│  AutoTokenizer.from_pretrained(model_name, local_files_only=True)  │
│  AutoModel.from_pretrained(model_name,     local_files_only=True)  │
│  model.eval()   ← inference only, no gradient computation          │
│                                                                     │
│  Note: use_fast=False is forced for mDeBERTa-v3 because its        │
│        SentencePiece tokeniser requires the protobuf backend.       │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 3 — Build Gender Geometry               [embeddings.py]       │
│  ─────────────────────────────────────────────────────────────────  │
│  For each anchor word w (male and female separately):               │
│                                                                     │
│  a) find_sentences(corpus, w, n=12, tokenizer)                      │
│       → tokenizer-level substring match (not string match)         │
│       → returns up to 12 sentences containing w                    │
│                                                                     │
│  b) word_vector_per_layer(model, tokenizer, sentences, w)           │
│       → for each sentence:                                          │
│           - run full forward pass, output_hidden_states=True        │
│           - locate w's subword tokens in the token list             │
│           - mean-pool those subword token vectors                   │
│             (handles multi-subword words like "ሰራሕተኛ ጽሬት")         │
│       → average all sentence vectors layer-by-layer                │
│       → result: one static vector per layer  (Bommasani et al. 2020)│
│                                                                     │
│  c) Aggregate across anchors → male centroid per layer              │
│                              → female centroid per layer            │
│                                                                     │
│  d) gender_direction[layer] = (male_centroid - female_centroid)     │
│                               ─────────────────────────────────    │
│                               ||male_centroid - female_centroid||   │
│       → unit vector pointing from female cluster to male cluster    │
│       → one direction vector per layer  (Bolukbasi et al. 2016)    │
└───────────────────────────────┬─────────────────────────────────────┘
                                │  gender geometry is computed ONCE
                                │  then reused for all 30 professions
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 4 — Score Each Profession               [metrics.py]          │
│  ─────────────────────────────────────────────────────────────────  │
│  For each profession term p:                                        │
│                                                                     │
│  a) find_sentences(corpus, p, n=20, tokenizer)                      │
│       → if fewer than 20 sentences found → SKIP this profession    │
│                                                                     │
│  b) word_vector_per_layer(model, tokenizer, sentences, p)           │
│       → same reduction as Step 3 but for profession terms          │
│       → result: s_l(p) — one static vector per layer               │
│                                                                     │
│  c) For each layer l, compute TWO independent estimators:           │
│                                                                     │
│     Estimator 1 — Projection Score (proj):                          │
│       ProjBias_l(p) = dot( s_l(p),  g_l )                          │
│       where g_l is the unit gender direction at layer l             │
│       + positive  → male-leaning                                    │
│       - negative  → female-leaning                                  │
│       Preserves vector magnitude; sensitive to association strength  │
│                                                                     │
│     Estimator 2 — Centroid Cosine Difference (cosdiff):             │
│       CosBias_l(p) = cos(s_l(p), c_M_l) - cos(s_l(p), c_F_l)      │
│       Pure angular measure; invariant to vector scale               │
│       Serves as cross-check for RQ3                                 │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 5 — Measure Estimator Agreement          [metrics.py]         │
│  ─────────────────────────────────────────────────────────────────  │
│  At each layer l, compute Spearman rank correlation                 │
│  between the two estimators across all profession terms:            │
│                                                                     │
│    ρ_l = Spearman( {ProjBias_l(p)}_p,  {CosBias_l(p)}_p )          │
│                                                                     │
│  ρ_l = 1.0  → both estimators rank professions in identical order   │
│  ρ_l = 0.0  → no agreement                                         │
│                                                                     │
│  Results across all models: ρ ≥ 0.970 at every valid layer         │
│  → confirms bias rankings are not an artifact of estimator choice   │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 6 — Aggregate and Save Outputs           [main.py, output.py] │
│  ─────────────────────────────────────────────────────────────────  │
│  Write ti_bias_by_layer.csv      ← per-profession, per-layer scores │
│  Write ti_spearman_by_layer.csv  ← ρ per layer                     │
│  Write ti_projection_layer_mean.csv ← mean proj per layer           │
│  Save  figs/ti_{slug}_projection_curve.png ← publication figure     │
│                                                                     │
│  Output directory: output/ti/{model_name.replace("/","_")}/         │
└─────────────────────────────────────────────────────────────────────┘
```

### Import / Dependency Graph

```
main.py
  ├── config.py          (settings — no imports from other modules)
  ├── data_loader.py     (loading + sentence search)
  │     └── (no local imports)
  ├── embeddings.py      (forward passes + gender geometry)
  │     └── data_loader  (find_sentences)
  ├── metrics.py         (bias estimators + Spearman)
  │     ├── data_loader  (find_sentences)
  │     └── embeddings   (word_vector_per_layer)
  ├── output.py          (Tee logger + plot)
  │     └── (no local imports)
  └── debug_mode.py      (verbose trace mode)
        ├── config
        ├── data_loader  (find_all_matching, classify_sentence)
        ├── embeddings   (build_gender_geometry, word_vector_per_layer)
        ├── metrics      (projection_score, centroid_cosine_diff)
        └── output       (Tee)
```

Imports flow **strictly downward** — no circular dependencies.

---

## 5. Source Code Map

| File | Lines | What it contains |
|------|------:|------------------|
| `config.py` | 33 | All user-editable settings: language, model list, mode, context counts |
| `data_loader.py` | 92 | `load_corpus`, `load_anchors`, `load_professions`; `find_sentences` (tokenizer-level match); `find_all_matching`; `classify_sentence` (gender marker detection) |
| `embeddings.py` | 122 | `word_vector_per_layer` (forward pass → per-layer averaged vectors); `build_gender_geometry` (centroids + unit direction) |
| `metrics.py` | 101 | `projection_score` (dot product); `centroid_cosine_diff` (angular); `bias_scores_for_word` (orchestrates scoring for one profession); `spearman_per_layer` |
| `output.py` | 65 | `Tee` (stdout mirror to log file); `plot_curve` (saves PNG + CSV) |
| `debug_mode.py` | 277 | `run_debug` + `_write_batch_debug`: full verbose trace for one profession across all corpus batches, with per-batch log files and summary index |
| `main.py` | 145 | `run_bulk` (loops over all professions); `run_one_model` (loads model, dispatches to mode); `main` (entry point, loops over all models) |

### Standalone Utility Scripts

| File | Purpose |
|------|---------|
| `check_rep.py` | Computes pairwise cosine similarity across all contextual vectors of a word at each layer — used to produce the representation uniqueness table in the paper (Section 4.1) |
| `target_weights.py` | Inspects the model's *static* (weight-level, non-contextual) representation of a target word — useful for distinguishing tokeniser-level vs contextual effects |
| `diagnsotic.py` | Tokeniser coverage check: for each profession and anchor term, reports how many corpus sentences are found under each model's vocabulary |

### Corpus and Anchor Generation Scripts

| File | Purpose |
|------|---------|
| `tigrigna_corpus_builder.py` | Generated the initial Tigrigna profession corpus (v1, template-based) |
| `tigrigna_corpus_builder_v2.py` | Refined corpus builder (v2) — currently active data |
| `tigrigna_anchor_builder.py` | Generated `anchors_ti.csv` from a curated list of Tigrigna gender markers |
| `prepare_tigrigna_corpus.py` | Pre-processed raw corpus text (whitespace normalisation, deduplication) |

These scripts were run once to produce the data files in `data/tigrigna/`.
They are kept for reproducibility — do not delete them.

---

## 6. Configuration Reference

All settings live in `src/config.py`. Nothing else needs to be edited to run experiments.

```python
# Which language to analyse
LANGUAGE = "ti"          # "ti" = Tigrigna | "ar" = Arabic | "es" = Spanish

# Which models to run — can be a list (run all) or a single string (run one)
MODEL_NAMES = [
    "xlm-roberta-base",
    "xlm-roberta-large",
    "facebook/xlm-v-base",
    "microsoft/mdeberta-v3-base",
]
# To run a single model, replace with:
# MODEL_NAMES = "xlm-roberta-base"

# Slug map: short names used in output filenames and paper figure paths
MODEL_SLUG_MAP = {
    "bert-base-multilingual-cased": "mbert",
    "xlm-roberta-base":             "xlmr_base",
    "xlm-roberta-large":            "xlmr_large",
    "facebook/xlm-v-base":          "xlmv_base",
    "microsoft/mdeberta-v3-base":   "mdeberta",
}

# Run mode
MODE = "bulk"            # "bulk"  = score all professions (paper results)
                         # "debug" = verbose single-word trace (development)

# Which word to trace in debug mode
DEBUG_WORD = "ሓረስታይ"

# How many corpus sentences to use per word
ANCHOR_CONTEXTS     = 12   # per anchor word (gender geometry construction)
PROFESSION_CONTEXTS = 20   # per profession term (bias scoring)
```

**Context count guidance:**

| PROFESSION_CONTEXTS | Use case | Stability |
|--------------------:|----------|-----------|
| 1–5 | Quick sanity check only | Unreliable |
| 10–20 | Standard paper results | Good |
| 50+ | Maximum stability | Slow on CPU |

---

## 7. How to Run

### Prerequisites

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# Install dependencies
pip install torch transformers scipy matplotlib sentencepiece protobuf tiktoken
```

mDeBERTa-v3-base specifically requires `sentencepiece`, `protobuf`, and `tiktoken`.
All other models only need `torch` and `transformers`.

### Step 1 — Configure

Edit `src/config.py`:

```python
LANGUAGE    = "ti"                  # language to analyse
MODEL_NAMES = ["xlm-roberta-base"]  # model(s) to run
MODE        = "bulk"                # "bulk" for paper results
```

### Step 2 — Run

```bash
python src/main.py
```

All outputs are written automatically to `output/{lang}/{model_name}/`.

### Step 3 — Inspect outputs

```
output/ti/xlm-roberta-base/
├── ti_bias_by_layer.csv              ← main results
├── ti_spearman_by_layer.csv          ← estimator agreement
├── ti_projection_layer_mean.csv      ← layer-averaged curve data
└── figs/
    └── ti_xlmr_base_projection_curve.png
```

### Running in debug mode (single profession, full trace)

```python
# In config.py:
MODE       = "debug"
DEBUG_WORD = "ሓኪም"       # any profession term from professions_ti.csv
```

```bash
python src/main.py
```

Debug output is written to `output/ti/{word}/`:
- `debug_{word}_index.txt` — batch map + summary statistics table
- `debug_{word}_batch00.txt`, `batch01.txt`, ... — one log file per batch

Each batch log contains: corpus overview, anchor coverage check, gender geometry
separation norms, all 20 context sentences with gender labels (M/F/N), per-layer
bias scores, and a final verdict with consistency analysis.

---

## 8. Output Files Explained

### `{lang}_bias_by_layer.csv` — Primary results

One row per (profession × layer) combination.

| Column | Type | Description |
|--------|------|-------------|
| `language` | str | Language code (`ti`, `ar`, `es`) |
| `model` | str | Full HuggingFace model name |
| `term` | str | Profession term in the target language |
| `layer` | int | Transformer layer index. Layer 0 = embedding layer (no self-attention). Layer N = final transformer block. |
| `proj` | float | **Projection score** — dot product of profession vector with the unit gender direction. Positive = male-leaning. Negative = female-leaning. Zero = equidistant from both gender poles. |
| `cosdiff` | float | **Cosine difference** — `cosine(profession, male_centroid) − cosine(profession, female_centroid)`. Pure angular measure; scale-invariant. Second independent estimator used to validate `proj`. |

**What to look for:**
- A word with `proj > 0` consistently across layers → robust male association
- Bias near zero at layer 0 but growing toward layer 9–11 → association is built during contextual processing, not inherited from tokenisation alone
- Agreement between `proj` and `cosdiff` at the same layer → result is robust

### `{lang}_spearman_by_layer.csv` — Estimator agreement (RQ3)

| Column | Type | Description |
|--------|------|-------------|
| `language` | str | Language code |
| `model` | str | Model name |
| `layer` | int | Transformer layer |
| `spearman_rho` | float | Rank correlation between `proj` and `cosdiff` across all profession terms at this layer. Range: −1 to +1. Values ≥ 0.970 indicate near-perfect ordinal agreement. |
| `spearman_p` | float | p-value for the correlation. Values < 0.05 indicate statistical significance. |
| `n` | int | Number of profession terms used (excluded terms that were skipped) |

### `{lang}_projection_layer_mean.csv` — Layer curve data

| Column | Type | Description |
|--------|------|-------------|
| `layer` | int | Transformer layer |
| `mean_proj` | float | Mean projection score across all scored profession terms at this layer. Positive = overall male-leaning; negative = overall female-leaning. This is what the PNG figure plots. |

### `figs/{lang}_{model_slug}_projection_curve.png` — Layer trajectory figure

A line chart with:
- **X-axis**: transformer layer (0 = embedding, N = final)
- **Y-axis**: mean projection score across all professions
- **Dashed grey line at y = 0**: neutral baseline (equal distance to both gender poles)

Reading the chart:
- Line **above zero** → model leans male on average at that depth
- Line **below zero** → model leans female
- **Rising slope** in mid-layers → bias accumulates during contextual processing
- **Peak** at layer 9–11 (base) or 23 (large) → mid-to-upper semantic layers encode strongest associations
- **Collapse** at the final layer → final hidden states are influenced by the language model head's prediction objective, which partially washes out profession-specific associations

---

## 9. Five-Model Results Summary

All results are for Tigrigna (`ti`). Peak layer = layer with highest mean projection.
M:F:N = professions majority male-leaning : female-leaning : neutral across all layers.

| Model | Layers | Vocab | Peak layer | Peak proj | M:F:N | Min ρ |
|-------|:------:|-------|:----------:|----------:|-------|------:|
| mBERT | 13 | WordPiece ~120K | — | −3.61 *(degenerate)* | n/a | n/a |
| XLM-R-base | 13 | SentencePiece 250K | 9 | +2.99 | 26:3:1 | 0.992 |
| XLM-R-large | 25 | SentencePiece 250K | 23 | +3.46 | 25:3:2 | 0.992 |
| XLM-V-base | 13 | SentencePiece 1M | 11 | +2.63 | 25:2:1† | 0.970‡ |
| mDeBERTa-v3-base | 13 | SentencePiece ~128K | 11 | −1.84 *(reversed)* | 2:28:0 | 0.974 |

**† XLM-V:** 28/30 professions scored; ጠበቓ (lawyer) and ነዳቓይ (mechanic) skipped —
0 corpus hits because the 1M vocabulary tokenises them differently from the 250K vocabulary
used to generate the corpus.

**‡ XLM-V layer 12** has ρ = 0.720 (representational collapse); all other layers ρ ≥ 0.970.

**mBERT:** All Tigrigna characters map to `[UNK]` under WordPiece — 26 single-word
professions are identical at every layer; 4 multi-word professions form a second cluster.
No profession-level differentiation exists. mBERT is a **negative control**, not a valid result.

**mDeBERTa reversal:** The only systematic architectural difference from XLM-R is
Disentangled Attention (separate content and position attention matrices). This reorganises
the internal geometry such that profession vectors cluster on the *female* side of the gender
axis — not because the model associates professions with femininity culturally, but because
its representational geometry is structured differently.

---

## 10. Run Modes

### Bulk mode (`MODE = "bulk"`)

Scores all 30 professions. One progress line per profession. Writes all four output files.
This is the mode used to produce the paper results.

```
==============================
  BULK MODE  |  language=ti  |  30 professions
==============================

Building gender geometry...
  Gender direction built across 13 layers

Scoring professions...
  [ 1/30] ሓኪም                     OK  mean_proj=+0.626  -> male
  [ 2/30] ነርስ                      OK  mean_proj=+0.441  -> male
  [ 3/30] ጠበቓ                     SKIPPED (0/20 sentences)
  ...
```

### Debug mode (`MODE = "debug"`)

Full verbose trace for one profession (`DEBUG_WORD`). Splits all available corpus
sentences into batches of `PROFESSION_CONTEXTS`, scores each batch independently,
and writes separate log files per batch.

Use this to:
- Verify the pipeline is working correctly for a new language
- Inspect which corpus sentences are being selected
- Check gender balance of the context sentences (M/F/N classification)
- Understand why a word might be skipped

```
output/ti/ሓረስታይ/
├── debug_ሓረስታይ_index.txt        ← summary of all batches
├── debug_ሓረስታይ_batch00.txt      ← full trace for sentences 0–19
├── debug_ሓረስታይ_batch01.txt      ← full trace for sentences 20–39
└── ...
```

---

## 11. Design Decisions

### Why tokenizer-level sentence matching?

`find_sentences()` uses the model's tokenizer to find sentences, not Python's `in` operator.
This ensures that a profession term is only counted as "present" if the tokenizer produces
the same subword sequence in both the word and the sentence — the same condition that must
hold for the embedding extraction step to succeed.

String matching (`word in sentence`) can produce false positives when a word appears as
part of a longer word with different tokenisation.

### Why two bias estimators?

**Projection score** (`proj`) preserves vector magnitude — it is sensitive to how far
a profession sits along the gender axis, capturing both direction and distance.

**Cosine difference** (`cosdiff`) is scale-invariant — it captures only the angular
relationship to each gender pole, ignoring magnitude.

Using both and measuring their Spearman rank correlation at each layer (RQ3) confirms
that the observed bias rankings are a property of the model's geometry, not an artifact
of the chosen estimator.

### Why mean-pooling subwords?

Words like ሰራሕተኛ ጽሬት (cleaning worker) tokenise to multiple subword pieces.
Mean-pooling across all subword token vectors produces a single representation for
the word, which can then be compared to single-token words on the same scale.

### Why is the gender direction re-normalised?

After computing `direction = male_centroid − female_centroid`, the vector is divided
by its own norm (`direction / ||direction||`). This makes the projection score
interpretable as a signed distance along a unit axis, comparable across layers
and models regardless of how far apart the two centroids happen to be.

### Why no GPU / no fine-tuning?

The analysis is entirely geometric — no training signal is needed. All models are
loaded in inference mode (`model.eval()`), gradients are disabled (`torch.no_grad()`),
and no parameter updates are performed. This makes the pipeline fully reproducible
on any machine.

### Why `use_fast=False` for mDeBERTa?

`microsoft/mdeberta-v3-base` uses a SentencePiece tokeniser that requires the
`protobuf` backend. The fast (Rust-based) tokenizer does not support this. Setting
`use_fast=False` forces the Python tokenizer, which handles the protobuf model correctly.
This is detected automatically in `run_one_model()`.

---

## 12. Dependencies

| Package | Role |
|---------|------|
| `torch` | Tensor operations; model inference |
| `transformers` | Load HuggingFace model checkpoints |
| `scipy` | `spearmanr` for estimator agreement |
| `matplotlib` | Projection curve PNG |
| `sentencepiece` | Required by mDeBERTa-v3 tokenizer |
| `protobuf` | Required by mDeBERTa-v3 tokenizer |
| `tiktoken` | Required by mDeBERTa-v3 tokenizer |

All models are loaded from the local `.models/hub/` cache (`local_files_only=True`).
No internet connection is required at runtime.
