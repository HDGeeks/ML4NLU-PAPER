"""
Multilingual Gender Bias Detector
----------------------------------
Measures how much gender bias exists in the way a language model represents
different job titles (e.g. "nurse", "engineer").

USAGE:
  python src/main.py

  All settings are controlled by the global variables in the Settings block
  below. No command-line arguments needed — just edit and run.
"""

import os
import csv
import logging
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModel

logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"


# ══════════════════════════════════════════════════════════════════════════════
#  Settings
# ══════════════════════════════════════════════════════════════════════════════

# ── Language & model ──────────────────────────────────────────────────────────
LANGUAGE   = "ti"                          # "en" | "es" | "ar" | "ti"
#MODEL_NAME = "bert-base-multilingual-cased"
MODEL_NAME = "xlm-roberta-base"          # ← switch model here

# ── Run mode ───────────────────────────────────────────────────────────────────
# "debug" → single profession, full verbose trace — use to understand the pipeline
# "bulk"  → all professions, progress line per word — use for paper results
MODE = "debug"

# ── Debug word (only used when MODE = "debug") ────────────────────────────────
# Set to any profession term from your professions CSV.
DEBUG_WORD = "ሓኪም"

# ── Context counts ─────────────────────────────────────────────────────────────
ANCHOR_CONTEXTS     = 3    # sentences per anchor word   (paper: 3)
PROFESSION_CONTEXTS = 5   # sentences per profession    (paper: 10)


# ══════════════════════════════════════════════════════════════════════════════
#  Core pipeline functions
# ══════════════════════════════════════════════════════════════════════════════

def find_sentences(corpus, word, n, tokenizer, verbose=False):
    """
    Return up to n sentences from the corpus that contain the given word.
    Uses tokenizer-level matching so subword splits are handled correctly.
    """
    word_tokens = tokenizer.tokenize(word)

    if verbose:
        print(f"\n  find_sentences('{word}', n={n})")
        print(f"    word tokenizes to: {word_tokens}")

    found = []
    for sentence in corpus:
        sentence_tokens = tokenizer.tokenize(sentence)
        for i in range(len(sentence_tokens) - len(word_tokens) + 1):
            if sentence_tokens[i : i + len(word_tokens)] == word_tokens:
                found.append(sentence)
                break
        if len(found) >= n:
            break

    if verbose:
        print(f"    found {len(found)}/{n} sentences")
        for i, s in enumerate(found):
            print(f"    [{i+1}] {s}")
        if len(found) < n:
            print(f"    ⚠ Only {len(found)} sentences found — word will be SKIPPED in bulk mode")

    return found


def word_vector_per_layer(model, tokenizer, sentences, word, verbose=False):
    """
    Extract the hidden states for the target word's tokens in each sentence,
    then average across sentences. Returns one vector per transformer layer.
    """
    word_tokens = tokenizer.tokenize(word)

    if verbose:
        print(f"\n  word_vector_per_layer('{word}')")
        print(f"    Processing {len(sentences)} sentences across all layers…")

    layer_accumulators = None
    count = 0

    for sent_idx, sentence in enumerate(sentences):
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        sentence_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        word_position = None
        for i in range(len(sentence_tokens) - len(word_tokens) + 1):
            if sentence_tokens[i : i + len(word_tokens)] == word_tokens:
                word_position = (i, i + len(word_tokens))
                break

        if word_position is None:
            if verbose:
                print(f"    [sent {sent_idx+1}] word not found in token list — skipping")
            continue

        start, end = word_position

        if verbose:
            print(f"    [sent {sent_idx+1}] word found at token positions [{start}:{end}]")
            print(f"             tokens: {sentence_tokens[start:end]}")
            layer_0_vec = outputs.hidden_states[0][0, start:end].mean(dim=0)
            last_vec    = outputs.hidden_states[-1][0, start:end].mean(dim=0)
            print(f"             layer 0 vector norm:  {layer_0_vec.norm():.4f}")
            print(f"             last layer vec norm:  {last_vec.norm():.4f}")

        layer_vecs = [
            layer_hidden[0, start:end].mean(dim=0)
            for layer_hidden in outputs.hidden_states
        ]

        if layer_accumulators is None:
            layer_accumulators = layer_vecs
        else:
            layer_accumulators = [a + b for a, b in zip(layer_accumulators, layer_vecs)]
        count += 1

    if count == 0 or layer_accumulators is None:
        if verbose:
            print(f"    ✗ No valid sentences — cannot build vector for '{word}'")
        return None

    averaged = [v / count for v in layer_accumulators]

    if verbose:
        print(f"    ✓ Averaged across {count} sentences → {len(averaged)} layer vectors")
        print(f"      Sample (layer 6 norm): {averaged[6].norm():.4f}")

    return averaged


def build_gender_geometry(model, tokenizer, corpus, male_words, female_words,
                           n_contexts, verbose=False):
    """
    Build male centroid, female centroid, and unit gender direction per layer.
    """
    if verbose:
        print("\n" + "─"*60)
        print("STEP: Building gender geometry")
        print("─"*60)
        print(f"  Male anchors   ({len(male_words)}): {male_words}")
        print(f"  Female anchors ({len(female_words)}): {female_words}")
        print(f"  Context sentences per anchor: {n_contexts}")

    def centroid_for_words(words, gender_label):
        all_layer_vecs = None
        count = 0
        skipped = []

        for word in words:
            sents = find_sentences(corpus, word, n_contexts, tokenizer, verbose=verbose)
            if len(sents) < n_contexts:
                skipped.append((word, len(sents)))
                continue
            layer_vecs = word_vector_per_layer(model, tokenizer, sents, word, verbose=verbose)
            if layer_vecs is None:
                skipped.append((word, 0))
                continue
            if all_layer_vecs is None:
                all_layer_vecs = layer_vecs
            else:
                all_layer_vecs = [a + b for a, b in zip(all_layer_vecs, layer_vecs)]
            count += 1

        if verbose:
            print(f"\n  {gender_label} centroid: built from {count}/{len(words)} anchor words")
            if skipped:
                print(f"  Skipped anchors: {skipped}")

        if count == 0:
            raise ValueError(f"No {gender_label} anchor words had enough corpus sentences.")

        return [v / count for v in all_layer_vecs]

    male_centroids   = centroid_for_words(male_words,   "Male")
    female_centroids = centroid_for_words(female_words, "Female")

    directions = []
    for layer_idx, (m, f) in enumerate(zip(male_centroids, female_centroids)):
        diff = m - f
        norm = diff.norm()

        if norm < 1e-8:
            raise ValueError(
                f"Male and female centroids are identical at layer {layer_idx}. "
                "Check that your anchor words appear in the corpus."
            )

        unit = diff / norm
        assert abs(unit.norm().item() - 1.0) < 1e-5, \
            f"Layer {layer_idx}: gender direction is not unit length after normalization."
        directions.append(unit)

    if verbose:
        print(f"\n  ✓ Gender direction built for {len(directions)} layers")
        # Show separation magnitude at a few layers as a health check
        print("  Centroid separation (raw direction norm) per layer:")
        sample_layers = [0, 4, 8, 12] if len(directions) > 12 else list(range(len(directions)))
        for l in sample_layers:
            raw_norm = (male_centroids[l] - female_centroids[l]).norm()
            print(f"    Layer {l:2d}: ‖male − female‖ = {raw_norm:.4f}")
        print("  (A larger norm = male and female clusters further apart at that layer)")

    return male_centroids, female_centroids, directions


def projection_score(vec, direction):
    """
    Scalar projection of the profession vector onto the gender direction.
    Paper formula: ProjBias(p) = dot(s(p), g/‖g‖)
    vec is NOT normalized — see inline comment in bias_scores_for_word.
    """
    return float(torch.dot(vec, direction))


def centroid_cosine_diff(vec, male_centroid, female_centroid):
    """
    cosine(word, male_centroid) − cosine(word, female_centroid)
    Positive = closer to male cluster. Independent second estimator for RQ3.
    """
    def cosine(a, b):
        return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-8))
    return cosine(vec, male_centroid) - cosine(vec, female_centroid)


def bias_scores_for_word(model, tokenizer, corpus, word,
                          male_centroids, female_centroids, directions,
                          verbose=False):
    """
    Returns per-layer bias scores for a profession word, or (None, count) if
    the word doesn't appear enough times.
    """
    if verbose:
        print("\n" + "─"*60)
        print(f"STEP: Scoring profession word  '{word}'")
        print("─"*60)

    sentences = find_sentences(corpus, word, PROFESSION_CONTEXTS, tokenizer, verbose=verbose)
    if len(sentences) < PROFESSION_CONTEXTS:
        return None, len(sentences)

    layer_vecs = word_vector_per_layer(model, tokenizer, sentences, word, verbose=verbose)
    if layer_vecs is None:
        return None, 0

    scores = []
    if verbose:
        print(f"\n  Computing bias scores across {len(layer_vecs)} layers…")
        print(f"  {'Layer':>6}  {'proj':>10}  {'cosdiff':>10}  interpretation")
        print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*30}")

    for layer_idx, vec in enumerate(layer_vecs):
        proj    = projection_score(vec, directions[layer_idx])
        cosdiff = centroid_cosine_diff(vec, male_centroids[layer_idx], female_centroids[layer_idx])

        if verbose:
            if abs(proj) < 0.05:
                interp = "≈ neutral"
            elif proj > 0:
                interp = f"→ male-leaning  (+{proj:.3f})"
            else:
                interp = f"→ female-leaning ({proj:.3f})"
            print(f"  {layer_idx:>6}  {proj:>10.4f}  {cosdiff:>10.4f}  {interp}")

        scores.append({"layer": layer_idx, "proj": proj, "cosdiff": cosdiff})

    if verbose:
        projs = [s["proj"] for s in scores]
        print(f"\n  Summary for '{word}':")
        print(f"    Mean projection across all layers: {sum(projs)/len(projs):.4f}")
        print(f"    Most male-leaning layer:   {max(range(len(projs)), key=lambda i: projs[i])} "
              f"(proj={max(projs):.4f})")
        print(f"    Most female-leaning layer: {min(range(len(projs)), key=lambda i: projs[i])} "
              f"(proj={min(projs):.4f})")
        print(f"\n  NOTE: proj and cosdiff are intentionally DIFFERENT metrics.")
        print(f"  proj   = dot(word_vec, gender_direction)  — preserves vector magnitude")
        print(f"  cosdiff = cosine_to_male − cosine_to_female — pure angular measure")
        print(f"  Their Spearman correlation across all words tests RQ3 (metric agreement).")

    return scores, len(sentences)


def spearman_per_layer(all_records):
    from scipy.stats import spearmanr
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


def plot_curve(all_records, out_png, out_csv, title):
    import matplotlib.pyplot as plt
    by_layer = defaultdict(list)
    for r in all_records:
        by_layer[r["layer"]].append(r["proj"])
    layers = sorted(by_layer.keys())
    means  = [sum(by_layer[l]) / len(by_layer[l]) for l in layers]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "mean_proj"])
        for l, m in zip(layers, means):
            writer.writerow([l, m])
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


# ══════════════════════════════════════════════════════════════════════════════
#  MODE: DEBUG — single profession, full verbose trace
# ══════════════════════════════════════════════════════════════════════════════

def run_debug(lang, word, tokenizer, model, corpus,
              male_words, female_words, job_titles):

    print("\n" + "═"*60)
    print(f"  DEBUG MODE  |  language={lang}  |  word='{word}'")
    print("═"*60)

    # ── Corpus overview ───────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("STEP: Corpus overview")
    print(f"{'─'*60}")
    print(f"  Total sentences in corpus: {len(corpus)}")
    print(f"  Sample sentences (first 3):")
    for s in corpus[:3]:
        print(f"    {s}")

    word_count = sum(1 for s in corpus if word in s)
    print(f"\n  Raw string occurrences of '{word}' in corpus: {word_count}")
    print(f"  (tokenizer-level matching may differ slightly)")

    # ── Anchor overview ───────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("STEP: Anchor word overview")
    print(f"{'─'*60}")
    print(f"  Male anchors   ({len(male_words)}): {male_words[:6]}{'…' if len(male_words)>6 else ''}")
    print(f"  Female anchors ({len(female_words)}): {female_words[:6]}{'…' if len(female_words)>6 else ''}")
    print(f"  ANCHOR_CONTEXTS = {ANCHOR_CONTEXTS}  (need this many sentences per anchor)")
    print(f"\n  Quick corpus coverage for anchors (string match):")
    for mw, fw in zip(male_words[:4], female_words[:4]):
        mc = sum(1 for s in corpus if mw in s)
        fc = sum(1 for s in corpus if fw in s)
        ms = "✓" if mc >= ANCHOR_CONTEXTS else "✗"
        fs = "✓" if fc >= ANCHOR_CONTEXTS else "✗"
        print(f"    {ms} {mw:<15} {mc:>4} sentences  |  "
              f"{fs} {fw:<15} {fc:>4} sentences")
    if len(male_words) > 4:
        print(f"    … ({len(male_words)-4} more anchor pairs not shown)")

    # ── Build gender geometry (verbose) ───────────────────────────────────────
    print(f"\n{'─'*60}")
    print("STEP: Building gender geometry (verbose)")
    print(f"{'─'*60}")
    male_centroids, female_centroids, directions = build_gender_geometry(
        model, tokenizer, corpus, male_words, female_words,
        ANCHOR_CONTEXTS, verbose=True
    )

    # ── Score the single profession word (verbose) ────────────────────────────
    scores, found = bias_scores_for_word(
        model, tokenizer, corpus, word,
        male_centroids, female_centroids, directions,
        verbose=True
    )

    if scores is None:
        print(f"\n  ✗ '{word}' could not be scored.")
        print(f"    Found {found} sentences, need {PROFESSION_CONTEXTS}.")
        print(f"    → Run tigrigna_corpus_builder.py and verify '{word}' is in PROFESSIONS_TI")
        return

    # ── Where does this word rank among all professions? ──────────────────────
    print(f"\n{'─'*60}")
    print(f"STEP: Rank of '{word}' among all {len(job_titles)} profession terms")
    print(f"{'─'*60}")
    print("  (Scoring all other professions silently to produce ranking…)")

    all_mean_projs = {}
    for job in job_titles:
        s, _ = bias_scores_for_word(
            model, tokenizer, corpus, job,
            male_centroids, female_centroids, directions,
            verbose=False
        )
        if s:
            all_mean_projs[job] = sum(r["proj"] for r in s) / len(s)

    if all_mean_projs:
        ranked = sorted(all_mean_projs.items(), key=lambda x: x[1], reverse=True)
        rank = next((i+1 for i, (j, _) in enumerate(ranked) if j == word), None)
        word_score = all_mean_projs.get(word)

        print(f"\n  Mean projection scores (all {len(ranked)} scored professions):")
        print(f"  {'Rank':>5}  {'Term':<20}  {'Mean proj':>10}")
        print(f"  {'─'*5}  {'─'*20}  {'─'*10}")
        for i, (job, score) in enumerate(ranked):
            marker = " ◄ YOU ARE HERE" if job == word else ""
            print(f"  {i+1:>5}  {job:<20}  {score:>10.4f}{marker}")

        if rank:
            print(f"\n  '{word}' ranks #{rank} out of {len(ranked)} scored professions")
            if word_score > 0.05:
                print(f"  Interpretation: male-leaning (mean proj = {word_score:.4f})")
            elif word_score < -0.05:
                print(f"  Interpretation: female-leaning (mean proj = {word_score:.4f})")
            else:
                print(f"  Interpretation: roughly neutral (mean proj = {word_score:.4f})")

    print(f"\n{'═'*60}")
    print("  DEBUG RUN COMPLETE")
    print(f"{'═'*60}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  MODE: BULK — all professions, progress only
# ══════════════════════════════════════════════════════════════════════════════

def run_bulk(lang, tokenizer, model, corpus,
             male_words, female_words, job_titles):

    output_dir   = Path("output") / lang / MODEL_NAME.replace("/", "_")
    bias_csv     = output_dir / f"{lang}_bias_by_layer.csv"
    spearman_csv = output_dir / f"{lang}_spearman_by_layer.csv"
    mean_csv     = output_dir / f"{lang}_projection_layer_mean.csv"
    figure_png   = output_dir / "figs" / f"{lang}_projection_curve.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figs").mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  BULK MODE  |  language={lang}  |  {len(job_titles)} professions")
    print(f"{'═'*60}\n")

    print("Building gender geometry…")
    male_centroids, female_centroids, directions = build_gender_geometry(
        model, tokenizer, corpus, male_words, female_words,
        ANCHOR_CONTEXTS, verbose=False
    )
    print(f"  ✓ Gender direction built across {len(directions)} layers")

    print("\nScoring job titles…")
    all_records = []
    skipped     = []

    for idx, job in enumerate(job_titles):
        print(f"  [{idx+1:2d}/{len(job_titles)}] {job:<25}", end="  ")
        scores, found = bias_scores_for_word(
            model, tokenizer, corpus, job,
            male_centroids, female_centroids, directions,
            verbose=False
        )
        if scores is None:
            skipped.append((job, found))
            print(f"SKIPPED (found {found}/{PROFESSION_CONTEXTS} sentences)")
            continue
        for s in scores:
            all_records.append({"term": job, **s})
        mean_proj = sum(s["proj"] for s in scores) / len(scores)
        direction_str = "→ male" if mean_proj > 0.05 else ("→ female" if mean_proj < -0.05 else "≈ neutral")
        print(f"✓  mean_proj={mean_proj:+.4f}  {direction_str}")

    kept = len(job_titles) - len(skipped)

    # Save CSVs
    with open(bias_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["language", "model", "term", "layer", "proj", "cosdiff"])
        for r in all_records:
            writer.writerow([lang, MODEL_NAME, r["term"], r["layer"], r["proj"], r["cosdiff"]])

    spearman_rows = spearman_per_layer(all_records)
    with open(spearman_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["language", "model", "layer", "spearman_rho", "spearman_p", "n"])
        for r in spearman_rows:
            writer.writerow([lang, MODEL_NAME, r["layer"], r["rho"], r["p"], r["n"]])

    plot_curve(
        all_records,
        out_png=str(figure_png),
        out_csv=str(mean_csv),
        title=f"Layer-wise mean projection ({lang} | {MODEL_NAME})",
    )

    print(f"\n{'─'*60}")
    print(f"Done!  Results saved to: {output_dir}")
    print(f"  Professions scored:  {kept} / {len(job_titles)}")
    if skipped:
        print(f"  Skipped ({len(skipped)}):")
        for job, n in skipped:
            print(f"    - {job}: found {n} sentence(s), need {PROFESSION_CONTEXTS}")

    if spearman_rows:
        mid = len(spearman_rows) // 2
        rho = spearman_rows[mid]["rho"]
        print(f"\n  Spearman ρ at layer {spearman_rows[mid]['layer']} "
              f"(mid-network): {rho:.3f}  "
              f"({'strong' if abs(rho)>0.7 else 'moderate' if abs(rho)>0.4 else 'weak'} agreement)")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    from data_loader import load_corpus, load_anchors, load_professions
    corpus                   = load_corpus(LANGUAGE)
    male_words, female_words = load_anchors(LANGUAGE)
    job_titles               = load_professions(LANGUAGE)

    print("Loading model…")
    # local_files_only=True uses the cached model without any network call
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model     = AutoModel.from_pretrained(MODEL_NAME,     local_files_only=True)
    model.eval()

    if MODE == "debug":
        run_debug(LANGUAGE, DEBUG_WORD, tokenizer, model, corpus,
                  male_words, female_words, job_titles)
    elif MODE == "bulk":
        run_bulk(LANGUAGE, tokenizer, model, corpus,
                 male_words, female_words, job_titles)
    else:
        raise ValueError(f"Unknown MODE '{MODE}' — set to 'debug' or 'bulk' in settings")


if __name__ == "__main__":
    main()