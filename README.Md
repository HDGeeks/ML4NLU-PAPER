# Multilingual Bias in Foundation Models  
## Layer-Wise Cross-Lingual Analysis Pipeline

This project measures how much **gender bias** a multilingual AI language model has learned — and how that bias changes as the model processes language through its internal layers.

The pipeline is CPU-friendly, requires no model training, and is designed for academic experiments.

---

# Research Idea

We measure how gender-related bias evolves across transformer layers by:

- Reducing contextual embeddings to static layer-wise representations
- Constructing a gender direction from anchor terms
- Projecting profession embeddings onto that direction
- Aggregating results across professions
- Visualizing layer-wise bias trends

---

# Project Structure

```
src/
    main.py                    → Runs the full pipeline; all settings live here
    data_loader.py             → Loads corpus, anchors, professions
    reduction.py               → Contextual → static layer-wise embeddings
    bias_metrics.py            → Gender direction + projection score
    plot_layer_projection.py   → Aggregates + plots mean layer curve
```

---

# Pipeline Overview

```
┌──────────────────────────────────────────────┐
│ STEP 1: Load Data                            │
│  - Corpus (language-specific)                │
│  - Gender anchors                            │
│  - Profession list                           │
└──────────────────────────────────────────────┘
                ↓
┌──────────────────────────────────────────────┐
│ STEP 2: Load Transformer Model               │
│  - e.g. mBERT or XLM-R                       │
│  - Enable hidden state extraction             │
└──────────────────────────────────────────────┘
                ↓
┌──────────────────────────────────────────────┐
│ STEP 3: Contextual → Static Reduction        │
│  For each word:                              │
│    - Extract contextual token embeddings      │
│    - Mean-pool subwords                      │
│    - Average across N contexts               │
│    - Obtain layer-wise static vectors        │
└──────────────────────────────────────────────┘
                ↓
┌──────────────────────────────────────────────┐
│ STEP 4: Build Gender Direction               │
│  - Compute centroid(male anchors)            │
│  - Compute centroid(female anchors)          │
│  - Direction = male − female                 │
└──────────────────────────────────────────────┘
                ↓
┌──────────────────────────────────────────────┐
│ STEP 5: Compute Projection Bias              │
│  For each profession and each layer:         │
│    projection = dot(prof_vector, direction)  │
│  → Save to bias CSV                          │
└──────────────────────────────────────────────┘
                ↓
┌──────────────────────────────────────────────┐
│ STEP 6: Aggregate Across Professions         │
│  - Compute mean projection per layer         │
│  - Save layer-mean CSV                       │
└──────────────────────────────────────────────┘
                ↓
┌──────────────────────────────────────────────┐
│ STEP 7: Plot Layer-Wise Bias Curve           │
│  - Layer (x-axis)                            │
│  - Mean projection (y-axis)                  │
│  - Export publication-ready PNG              │
└──────────────────────────────────────────────┘
```

---

# How To Run

## 1) Set your language and model

Open `main.py` and edit the settings at the top:

```python
LANGUAGE   = "es"                        # "en", "es", "ar", or "ti"
MODEL_NAME = "bert-base-multilingual-cased"
```

## 2) Run the pipeline

```
python src/main.py
```

All outputs are saved automatically to `output/<language>/`.

---

# Understanding the Outputs

The pipeline produces four files. Here is what each one tells you.

---

## `{lang}_bias_by_layer.csv` — The main results

This is the most important file. It records, for every job title, how "gendered"
the model considers it to be — at each internal processing stage (layer).

| Column | What it means |
|--------|--------------|
| `language` | Language the analysis ran on (e.g. `en` for English) |
| `model` | Which AI model was used |
| `term` | The job title being measured (e.g. `nurse`, `engineer`) |
| `layer` | The model's processing stage. Think of layers like floors in a building — layer 0 is the ground floor (earliest, most basic processing), layer 12 is the top floor (deepest, most complex understanding) |
| `proj` | **Projection score** — the main bias number. Positive (+) means the model associates this word more with males. Negative (−) means more female association. Zero is neutral. A value of +0.8 for "engineer" would be a strong male association. |
| `cosdiff` | **Cosine difference** — a second way of measuring the same thing, for cross-checking. It asks: is this word sitting closer to the "male cluster" or the "female cluster" in the model's internal space? Positive = closer to male, negative = closer to female. If both `proj` and `cosdiff` agree, you can be more confident in the result. |

**What to look for:** If job titles like "nurse" consistently show negative scores and "engineer" consistently shows positive scores — across multiple layers — the model has learned real-world gender stereotypes from its training data.

---

## `{lang}_spearman_by_layer.csv` — The reliability check

This answers the question: *"Do our two bias measurements (`proj` and `cosdiff`) agree with each other?"*
If they both rank job titles in the same order (e.g. both say "surgeon" is more male-coded than "teacher"), that's reassuring.

| Column | What it means |
|--------|--------------|
| `language` | Language the analysis ran on |
| `model` | Which AI model was used |
| `layer` | Processing stage (same numbering as above) |
| `spearman_rho` | The agreement score, ranging from −1 to +1. Think of it like a grade: +1 means the two metrics perfectly agree, 0 means they're unrelated, −1 means they completely disagree. Anything above +0.7 is strong agreement. |
| `spearman_p` | Confidence level. A value below 0.05 means the agreement is statistically meaningful and not just coincidence. |
| `n` | How many job titles were used to compute this. More is better — a high agreement score based on only 3 words isn't very trustworthy. |

---

## `{lang}_projection_layer_mean.csv` — The data behind the chart

This is a simplified summary: just the average bias score across all job titles, per layer.
You don't need to read this directly — the figure below shows the same information visually.

| Column | What it means |
|--------|--------------|
| `layer` | Processing stage |
| `mean_proj` | The average projection score across all job titles at that layer. If this is consistently positive, the model leans male on average across all professions tested. |

---

## `figs/{lang}_projection_curve.png` — The chart

This is a line chart with **layers on the X-axis** and **average bias on the Y-axis**.
The dashed horizontal line at zero is the neutral baseline.

- **Line above zero** → the model, on average, associates professions with males at that processing stage
- **Line below zero** → the average lean is female
- **The shape matters** — if bias is low in early layers but grows in later ones, it means the model is *building in* gender associations as it processes language more deeply. This is more concerning than bias appearing early, because later layers are closer to how the model actually "thinks" about a word.
- **A flat line near zero** across all layers would mean the model is largely gender-neutral — the ideal outcome.

---

# Experimental Design

| Setting | Options |
|---------|---------|
| Languages | English (`en`), Spanish (`es`), Arabic (`ar`), Tigrigna (`ti`) |
| Models | mBERT, XLM-R, or any encoder with hidden states |
| Contexts per word | Adjustable in `main.py` |

## Dev Mode vs Paper Mode

| Mode | Contexts per word | Purpose |
|------|------------------|---------|
| Dev mode | 1–3 | Quickly validate the pipeline is working |
| Paper mode | 10–20+ | Stable, publishable statistical results |

---

# Research Contribution

This pipeline enables:

- Layer-wise gender bias analysis
- Cross-model comparison (mBERT vs XLM-R)
- Cross-lingual comparison (English vs Arabic vs Tigrigna)
- Metric extension (projection + cosine difference)

It is CPU-compatible and requires no fine-tuning.

---

# Status

- Core pipeline complete  
- Aggregation and plotting complete  
- Modular and reproducible  
- Ready for multi-model experiments