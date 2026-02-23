# Multilingual Bias in Foundation Models  
## Layer-Wise Cross-Lingual Analysis Pipeline

This project implements a clean, CPU-friendly research pipeline to measure
layer-wise gender bias in multilingual transformer models.

The pipeline is modular, reproducible, and designed for academic experiments.

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
    main.py                    → Generates profession-level projection CSV
    data_loader.py             → Loads corpus, anchors, professions
    reduction.py               → Contextual → static layer-wise embeddings
    bias_metrics.py            → Gender direction + projection score
    plot_layer_projection.py   → Aggregates + plots mean layer curve
```

---

# Pipeline Overview (Algorithm)

```
┌──────────────────────────────────────────────┐
│ STEP 1: Load Data                           │
│  - Corpus (language-specific)               │
│  - Gender anchors                           │
│  - Profession list                          │
└──────────────────────────────────────────────┘
                ↓
┌──────────────────────────────────────────────┐
│ STEP 2: Load Transformer Model              │
│  - e.g. mBERT or XLM-R                      │
│  - Enable hidden state extraction            │
└──────────────────────────────────────────────┘
                ↓
┌──────────────────────────────────────────────┐
│ STEP 3: Contextual → Static Reduction       │
│  For each word:                             │
│    - Extract contextual token embeddings     │
│    - Mean-pool subwords                     │
│    - Average across N contexts              │
│    - Obtain layer-wise static vectors       │
└──────────────────────────────────────────────┘
                ↓
┌──────────────────────────────────────────────┐
│ STEP 4: Build Gender Direction              │
│  - Compute centroid(male anchors)           │
│  - Compute centroid(female anchors)         │
│  - Direction = male − female                │
└──────────────────────────────────────────────┘
                ↓
┌──────────────────────────────────────────────┐
│ STEP 5: Compute Projection Bias             │
│  For each profession and each layer:        │
│    projection = dot(prof_vector, direction) │
│  → Save to projection CSV                   │
└──────────────────────────────────────────────┘
                ↓
┌──────────────────────────────────────────────┐
│ STEP 6: Aggregate Across Professions        │
│  - Compute mean projection per layer        │
│  - Save layer-mean CSV                      │
└──────────────────────────────────────────────┘
                ↓
┌──────────────────────────────────────────────┐
│ STEP 7: Plot Layer-Wise Bias Curve          │
│  - Layer (x-axis)                           │
│  - Mean projection (y-axis)                 │
│  - Export publication-ready PNG             │
└──────────────────────────────────────────────┘
```

---

# How To Run

## 1) Generate Projection CSV

Edit the CONFIG block inside `main.py`:

```python
LANG = "es"
MODEL_NAME = "bert-base-multilingual-cased"
```

Then run:

```
python src/main.py
```

Output:

```
output/<language>_<model>_projection.csv
```

---

## 2) Generate Layer-Wise Plot

Edit the CONFIG block inside:

```
src/plot_layer_projection.py
```

Then run:

```
python src/plot_layer_projection.py
```

Outputs:

- output/figs/<...>.png  
- output/<...>_layer_mean.csv  

---

# Experimental Design

The pipeline supports:

- Languages: Spanish (es), Arabic (ar), Tigrigna (ti)
- Models: mBERT, XLM-R (or any encoder model with hidden states)
- Adjustable context sampling size
- Optional single-word debug mode

All experimental variables are controlled in the CONFIG block of `main.py`.

---

# Dev Mode vs Paper Mode

| Mode        | Contexts per word | Purpose                     |
|------------|------------------|----------------------------|
| Dev mode   | 1–3              | Validate pipeline logic    |
| Paper mode | 10–20+           | Stable statistical signal  |

---

# Output Format

Projection CSV schema:

```
language,model,profession,layer,projection
```

Layer-mean CSV schema:

```
layer,mean_projection
```

---

# Research Contribution

This implementation enables:

- Layer-wise bias analysis
- Cross-model comparison
- Cross-lingual comparison
- Metric extension (e.g., centroid cosine)

It is CPU-compatible and requires no fine-tuning.

---

# Status

Core pipeline complete  
Aggregation and plotting complete  
Modular and reproducible  
Ready for multi-model experiments  