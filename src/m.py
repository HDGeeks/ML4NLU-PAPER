"""
Multilingual Gender Bias Detector
----------------------------------
This script measures how much gender bias exists in the way a language model
represents different job titles (e.g. "nurse", "engineer").

It works by:
  1. Loading sentences from a corpus and a list of job titles
  2. Loading a pretrained multilingual language model (mBERT)
  3. Figuring out where "male" and "female" concepts sit in the model's space
  4. Measuring how close each job title is to the male vs. female side
  5. Saving results to CSV files and a plot image

No training. CPU-only. Deterministic.
"""

import os
import csv
import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel

# Suppress noisy transformer logs
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"


# ── Settings ─────────────────────────────────────────────────────────────────

LANGUAGE   = "en"                        # language code: "en", "es", "ar", "ti"
MODEL_NAME = "bert-base-multilingual-cased"
# MODEL_NAME = "xlm-roberta-base"        # ← uncomment to switch to XLM-R

# ── Context counts (paper-aligned) ───────────────────────────────────────────
# We need example sentences for each word so the model can build a stable
# representation of it. But not all words need the same number of examples.
#
# ANCHOR words (he, she, man, woman…) are very common — 3 sentences is enough.
#
# PROFESSION words (nurse, engineer…) are rarer and are the actual subject of
# our bias measurement, so we use more sentences (10) to get a reliable result.
# Using only 3 sentences for professions would be like judging someone's
# personality from a single conversation — too noisy to trust.
#
# For quick testing/dev runs, you can temporarily lower both numbers to 1–2.
# For final paper results, keep PROFESSION_CONTEXTS at 10 or higher.

ANCHOR_CONTEXTS     = 3    # sentences per gender anchor word  (paper: 3)
PROFESSION_CONTEXTS = 10   # sentences per profession term     (paper: 10)

OUTPUT_DIR = Path("output") / LANGUAGE / MODEL_NAME.replace("/", "_")


# ── Output file paths ─────────────────────────────────────────────────────────

bias_csv     = OUTPUT_DIR / f"{LANGUAGE}_bias_by_layer.csv"
spearman_csv = OUTPUT_DIR / f"{LANGUAGE}_spearman_by_layer.csv"
mean_csv     = OUTPUT_DIR / f"{LANGUAGE}_projection_layer_mean.csv"
figure_png   = OUTPUT_DIR / "figs" / f"{LANGUAGE}_projection_curve.png"


# ── Step 1: Find sentences containing a given word ────────────────────────────

def find_sentences(corpus, word, n, tokenizer):
    """Return up to n sentences from the corpus that contain the given word."""
    word_tokens = tokenizer.tokenize(word)
    found = []
    for sentence in corpus:
        sentence_tokens = tokenizer.tokenize(sentence)
        # Check if word_tokens appears anywhere in sentence_tokens
        for i in range(len(sentence_tokens) - len(word_tokens) + 1):
            if sentence_tokens[i : i + len(word_tokens)] == word_tokens:
                found.append(sentence)
                break
        if len(found) >= n:
            break
    return found


# ── Step 2: Turn sentences into a single averaged vector (one per layer) ──────

def word_vector_per_layer(model, tokenizer, sentences, word):
    """
    For each sentence, extract the hidden states for the target word's tokens,
    then average them across sentences. Returns one vector per transformer layer.
    """
    word_tokens = tokenizer.tokenize(word)
    layer_accumulators = None
    count = 0

    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        sentence_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # Find where the word appears in the tokenized sentence
        word_position = None
        for i in range(len(sentence_tokens) - len(word_tokens) + 1):
            if sentence_tokens[i : i + len(word_tokens)] == word_tokens:
                word_position = (i, i + len(word_tokens))
                break

        if word_position is None:
            continue

        start, end = word_position
        # hidden_states: tuple of (n_layers+1) tensors, each [1, seq_len, hidden_size]
        layer_vecs = [
            layer_hidden[0, start:end].mean(dim=0)   # average over word tokens
            for layer_hidden in outputs.hidden_states
        ]

        if layer_accumulators is None:
            layer_accumulators = layer_vecs
        else:
            layer_accumulators = [a + b for a, b in zip(layer_accumulators, layer_vecs)]
        count += 1

    if count == 0 or layer_accumulators is None:
        return None

    # Average across all sentences
    return [v / count for v in layer_accumulators]


# ── Step 3: Build the "gender direction" per layer ────────────────────────────

def build_gender_geometry(model, tokenizer, corpus, male_words, female_words, n_contexts):
    """
    For each layer, compute:
      - male centroid:    average position of male anchor words in the model's space
      - female centroid:  average position of female anchor words in the model's space
      - gender direction: the arrow pointing from the female cluster to the male cluster
    """
    def centroid_for_words(words):
        all_layer_vecs = None
        count = 0
        for word in words:
            sentences = find_sentences(corpus, word, n_contexts, tokenizer)
            if len(sentences) < n_contexts:
                continue
            layer_vecs = word_vector_per_layer(model, tokenizer, sentences, word)
            if layer_vecs is None:
                continue
            if all_layer_vecs is None:
                all_layer_vecs = layer_vecs
            else:
                all_layer_vecs = [a + b for a, b in zip(all_layer_vecs, layer_vecs)]
            count += 1
        if count == 0:
            raise ValueError("No anchor words had enough contexts in corpus.")
        return [v / count for v in all_layer_vecs]

    male_centroids   = centroid_for_words(male_words)
    female_centroids = centroid_for_words(female_words)

    # ── Build and normalize the gender direction ──────────────────────────────
    #
    # The gender direction is the arrow from the female centroid to the male
    # centroid: male_position − female_position.
    #
    # WHY WE NORMALIZE (shrink it to length exactly 1):
    # Think of it like switching from an unmarked stick to a proper ruler.
    # The raw arrow between centroids has a different length at every layer —
    # some layers spread male and female words far apart, others keep them close.
    # If we used the raw arrow, a score of 0.5 at layer 3 would mean something
    # completely different from a score of 0.5 at layer 10, making the
    # layer-by-layer comparison meaningless.
    # By rescaling the arrow to length 1, every layer uses the same ruler,
    # so scores are directly comparable across layers — which is the whole
    # point of the layer-wise analysis in the paper.
    #
    # We also add two safety checks:
    #   1. If the two centroids land in the exact same spot (norm ≈ 0), the
    #      direction is undefined — we stop immediately with a clear message.
    #   2. After dividing, we confirm the result really is length 1. If it
    #      isn't, something went wrong numerically and we want to know now,
    #      not after running the full experiment.

    directions = []
    for m, f in zip(male_centroids, female_centroids):
        diff = m - f
        norm = diff.norm()

        if norm < 1e-8:
            raise ValueError(
                "Male and female centroids are identical at one layer — "
                "check that your anchor words appear in the corpus."
            )

        unit = diff / norm  # ← rescale the arrow to length 1 (the "ruler")

        # Confirm the arrow is truly length 1 after rescaling.
        assert abs(unit.norm().item() - 1.0) < 1e-5, \
            "Gender direction is not a unit vector — normalization failed."

        directions.append(unit)

    return male_centroids, female_centroids, directions


# ── Step 4: Compute bias scores for a job title ───────────────────────────────

def projection_score(vec, direction):
    """
    Measures how far a job title's vector leans toward the male side.

    This implements the paper's main bias formula:
        ProjBias(p) = dot( s(p), g / ‖g‖ )

    Since 'direction' is already length 1, this simply becomes:
        ProjBias(p) = dot( s(p), direction )

    WHY WE DO NOT NORMALIZE the profession vector here:
    The previous version of this code accidentally normalized the profession
    vector before the dot product — which silently turned this into a cosine
    similarity. That was a bug, for two reasons:

    1. It no longer matched the paper's formula.

    2. It made the two bias metrics (proj and cosdiff) measure nearly the same
       thing. The whole point of running both is to cross-check them: if they
       agree, our result is more trustworthy. But if proj is also a cosine
       similarity — just like cosdiff — they will always agree, and the
       Spearman check (RQ3) becomes circular and meaningless.

    Leaving the profession vector at its natural length keeps the two metrics
    genuinely independent, so their agreement (or disagreement) actually tells
    us something about the robustness of the bias measurement.
    """
    return float(torch.dot(vec, direction))


def centroid_cosine_diff(vec, male_centroid, female_centroid):
    """
    A second, independent way to measure bias:
    how much closer is this word to the male cluster than the female cluster?
    (positive = closer to male side, negative = closer to female side)
    """
    def cosine(a, b):
        return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-8))
    return cosine(vec, male_centroid) - cosine(vec, female_centroid)


def bias_scores_for_word(model, tokenizer, corpus, word,
                          male_centroids, female_centroids, directions):
    """
    Returns a list of per-layer bias scores for a given job title,
    or None if the word doesn't appear enough times in the corpus.
    """
    sentences = find_sentences(corpus, word, PROFESSION_CONTEXTS, tokenizer)
    if len(sentences) < PROFESSION_CONTEXTS:
        return None, len(sentences)

    layer_vecs = word_vector_per_layer(model, tokenizer, sentences, word)
    if layer_vecs is None:
        return None, 0

    scores = []
    for layer_idx, vec in enumerate(layer_vecs):
        scores.append({
            "layer":   layer_idx,
            "proj":    projection_score(vec, directions[layer_idx]),
            "cosdiff": centroid_cosine_diff(vec, male_centroids[layer_idx], female_centroids[layer_idx]),
        })
    return scores, len(sentences)


# ── Step 5: Compute Spearman correlation between the two bias metrics ─────────

def spearman_per_layer(all_records):
    """
    For each layer, check whether 'proj' and 'cosdiff' agree on word rankings.
    A high Spearman correlation means the two metrics tell the same story.
    """
    from scipy.stats import spearmanr
    from collections import defaultdict

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


# ── Step 6: Plot the layer-wise mean projection curve ────────────────────────

def plot_curve(all_records, out_png, out_csv, title):
    """Save a line chart of mean bias projection per transformer layer."""
    import matplotlib.pyplot as plt
    from collections import defaultdict

    by_layer = defaultdict(list)
    for r in all_records:
        by_layer[r["layer"]].append(r["proj"])

    layers = sorted(by_layer.keys())
    means  = [sum(by_layer[l]) / len(by_layer[l]) for l in layers]

    # Save mean CSV
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "mean_proj"])
        for l, m in zip(layers, means):
            writer.writerow([l, m])

    # Save figure
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(layers, means, marker="o", linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Mean Projection Score\n(+ = male, − = female)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ── Main: wire everything together ────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "figs").mkdir(parents=True, exist_ok=True)

    # Load data (these functions live in your data_loader module)
    from data_loader import load_corpus, load_anchors, load_professions
    corpus                   = load_corpus(LANGUAGE)
    male_words, female_words = load_anchors(LANGUAGE)
    job_titles               = load_professions(LANGUAGE)

    # Load the language model
    print("Loading model…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    # Build the male/female geometry for every transformer layer.
    # Anchors use the smaller context count — they are common words, easy to find.
    print("Building gender geometry…")
    with torch.no_grad():
        male_centroids, female_centroids, directions = build_gender_geometry(
            model, tokenizer, corpus, male_words, female_words, ANCHOR_CONTEXTS
        )

    # Score every job title using the larger profession context count.
    print("Scoring job titles…")
    all_records = []
    skipped     = []

    for job in job_titles:
        scores, found = bias_scores_for_word(
            model, tokenizer, corpus, job,
            male_centroids, female_centroids, directions
        )
        if scores is None:
            skipped.append((job, found))
            continue
        for s in scores:
            all_records.append({"term": job, **s})

    kept = len(job_titles) - len(skipped)

    # Save bias scores CSV
    with open(bias_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["language", "model", "term", "layer", "proj", "cosdiff"])
        for r in all_records:
            writer.writerow([LANGUAGE, MODEL_NAME, r["term"], r["layer"], r["proj"], r["cosdiff"]])

    # Save Spearman agreement CSV
    spearman_rows = spearman_per_layer(all_records)
    with open(spearman_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["language", "model", "layer", "spearman_rho", "spearman_p", "n"])
        for r in spearman_rows:
            writer.writerow([LANGUAGE, MODEL_NAME, r["layer"], r["rho"], r["p"], r["n"]])

    # Save plot and layer-mean CSV
    plot_curve(
        all_records,
        out_png=str(figure_png),
        out_csv=str(mean_csv),
        title=f"Layer-wise mean projection ({LANGUAGE} | {MODEL_NAME})",
    )

    # Summary
    print(f"\nDone! Results saved to: {OUTPUT_DIR}")
    print(f"  Job titles scored:  {kept} / {len(job_titles)}")
    if skipped:
        print("  Skipped (not enough sentences in corpus):")
        for job, n in skipped:
            print(f"    - {job}: found {n} sentence(s), need {PROFESSION_CONTEXTS}")


if __name__ == "__main__":
    main()