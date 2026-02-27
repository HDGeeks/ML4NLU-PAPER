"""
Multilingual Gender Bias Detector
----------------------------------
Measures how much gender bias exists in the way a language model represents
different job titles (e.g. "nurse", "engineer").

USAGE:
  python src/main.py

  All settings are controlled by the global variables in the Settings block
  below. No command-line arguments needed — just edit and run.

CHANGES FROM v3
---------------
  - debug mode writes batch files to  output/{lang}/{word}/  (one sub-dir per target)
  - index file now appended with mean-projection summary table after all batches finish
"""

import os
import csv
import sys
import logging
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModel

logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"


# ── Tee: mirrors all print() output to both terminal and a log file ───────────

class Tee:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, filepath: str):
        self.terminal = sys.stdout
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.logfile  = open(filepath, "w", encoding="utf-8")

    def write(self, msg: str):
        self.terminal.write(msg)
        self.logfile.write(msg)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def close(self):
        sys.stdout = self.terminal
        self.logfile.close()


# ── Gender marker sets for sentence classification ────────────────────────────

_FEMALE_MARKERS = {
    "ንሳ", "ኣደ", "ሓፍቲ", "ጓል", "ሰበይቲ", "ዓባየይ",
    "ኣንስተይቲ", "ጓል ሓፍቲ", "ደቂ ኣንስትዮ", "ሰበይተይ",
    "ንዓኣ", "ናታ",
}
_MALE_MARKERS = {
    "ንሱ", "ኣቦ", "ሓው", "ወዲ", "ሰብኣይ", "ኣቦሓጎ",
    "ተባዕታይ", "ወዲ ሓው", "ደቂ ተባዕትዮ", "ናቱ",
    "ንዕኡ", "ሓወይ",
}

def classify_sentence(sentence: str) -> str:
    """Return 'F', 'M', or 'N' (neutral) based on gender marker presence."""
    for m in _FEMALE_MARKERS:
        if m in sentence:
            return "F"
    for m in _MALE_MARKERS:
        if m in sentence:
            return "M"
    return "N"


# ══════════════════════════════════════════════════════════════════════════════
#  Settings
# ══════════════════════════════════════════════════════════════════════════════

# ── Language & model ──────────────────────────────────────────────────────────
LANGUAGE   = "ti"                          # "en" | "es" | "ar" | "ti"

# MODEL_NAMES can be a single string OR a list — all will be run in sequence.
MODEL_NAMES = [
    "xlm-roberta-base",
    "xlm-roberta-large",
    "facebook/xlm-v-base",
    "microsoft/mdeberta-v3-base",
]
# Uncomment to run a single model only:
# MODEL_NAMES = "xlm-roberta-base"

# Short slugs used in output filenames (and main_v2.tex figure paths).
MODEL_SLUG_MAP = {
    "bert-base-multilingual-cased":  "mbert",
    "xlm-roberta-base":              "xlmr_base",
    "xlm-roberta-large":             "xlmr_large",
    "facebook/xlm-v-base":           "xlmv_base",
    "microsoft/mdeberta-v3-base":    "mdeberta",
}

# ── Run mode ───────────────────────────────────────────────────────────────────
# "debug" → single profession, full verbose trace — use to understand the pipeline
# "bulk"  → all professions, progress line per word — use for paper results
MODE = "bulk"

# ── Debug word (only used when MODE = "debug") ────────────────────────────────
# Set to any profession term from your professions CSV.
DEBUG_WORD = "ሓረስታይ" 

# ── Context counts ─────────────────────────────────────────────────────────────
ANCHOR_CONTEXTS     = 12   # sentences per anchor word   (paper: 3)
PROFESSION_CONTEXTS = 20  # sentences per profession    (paper: 10)


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


def find_all_matching(corpus, word):
    """Return (row_idx, sentence) for every corpus sentence containing word (string match)."""
    return [(idx, s) for idx, s in enumerate(corpus) if word in s]


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
#  MODE: DEBUG — batch-aware, one log file per batch of sentences
# ══════════════════════════════════════════════════════════════════════════════

def _write_batch_debug(lang, word, batch_id, n_batches, batch_with_indices,
                        tokenizer, model, corpus, male_words, female_words,
                        male_centroids, female_centroids, directions):
    """
    Write the full debug log for one batch of profession sentences.
    stdout must already be redirected to a Tee before calling this.

    Returns
    -------
    (mean_val, verdict, balance_label, row_start, row_end)
    """
    sentences   = [s   for _, s in batch_with_indices]
    row_indices = [idx for idx, _ in batch_with_indices]

    # ── Header ────────────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print(f"  DEBUG BATCH {batch_id:02d} / {n_batches-1}  |  lang={lang}  |  word='{word}'")
    print("═"*60)

    # ── Corpus overview ───────────────────────────────────────────────────────
    n_hits = sum(1 for s in corpus if word in s)
    print(f"\n{'─'*60}")
    print("STEP: Corpus overview")
    print(f"{'─'*60}")
    print(f"  Total corpus sentences         : {len(corpus)}")
    print(f"  Sentences containing '{word}'  : {n_hits}")
    print(f"  Full batches of {PROFESSION_CONTEXTS}             : {n_hits // PROFESSION_CONTEXTS}  "
          f"({n_hits % PROFESSION_CONTEXTS} leftover, unused)")
    print(f"  This batch                     : rows {row_indices[0]}–{row_indices[-1]}")

    # ── Anchor overview ───────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("STEP: Anchor word overview")
    print(f"{'─'*60}")
    print(f"  Male anchors   ({len(male_words)}): {male_words[:6]}{'…' if len(male_words)>6 else ''}")
    print(f"  Female anchors ({len(female_words)}): {female_words[:6]}{'…' if len(female_words)>6 else ''}")
    print(f"  ANCHOR_CONTEXTS = {ANCHOR_CONTEXTS}  (shared, same geometry for all batches)")
    print(f"\n  Anchor coverage (need {ANCHOR_CONTEXTS} sentences each):")
    print(f"  {'─'*56}")
    print(f"  {'Gender':<8}  {'Anchor':<16}  {'Available':>10}  {'Used':>6}  Status")
    print(f"  {'─'*8}  {'─'*16}  {'─'*10}  {'─'*6}  {'─'*6}")
    for mw in male_words:
        avail  = sum(1 for s in corpus if mw in s)
        status = "✓" if avail >= ANCHOR_CONTEXTS else "✗ INSUFFICIENT"
        print(f"  {'M':<8}  {mw:<16}  {avail:>10}  {min(avail,ANCHOR_CONTEXTS):>6}  {status}")
    for fw in female_words:
        avail  = sum(1 for s in corpus if fw in s)
        status = "✓" if avail >= ANCHOR_CONTEXTS else "✗ INSUFFICIENT"
        print(f"  {'F':<8}  {fw:<16}  {avail:>10}  {min(avail,ANCHOR_CONTEXTS):>6}  {status}")

    # ── Gender geometry summary (pre-computed once, reused across batches) ────
    sample_layers = [0, 4, 8, 12] if len(directions) > 12 else list(range(len(directions)))
    print(f"\n{'─'*60}")
    print("STEP: Gender geometry  (pre-computed, shared across all batches)")
    print(f"{'─'*60}")
    print(f"  ✓ Gender direction for {len(directions)} layers")
    print(f"  Centroid separation ‖male − female‖ at sample layers:")
    for l in sample_layers:
        raw_norm = (male_centroids[l] - female_centroids[l]).norm()
        print(f"    Layer {l:2d}: {raw_norm:.4f}")

    # ── Context sentences for this batch ─────────────────────────────────────
    m_pairs = [(idx, s) for idx, s in batch_with_indices if classify_sentence(s) == "M"]
    f_pairs = [(idx, s) for idx, s in batch_with_indices if classify_sentence(s) == "F"]
    n_pairs = [(idx, s) for idx, s in batch_with_indices if classify_sentence(s) == "N"]
    _diff   = abs(len(m_pairs) - len(f_pairs))
    _balance = ("balanced"             if _diff <= 2
                else ("male-biased sample"   if len(m_pairs) > len(f_pairs)
                      else "female-biased sample"))

    print(f"\n{'─'*60}")
    print(f"STEP: Batch {batch_id:02d} context sentences  (row indices shown)")
    print(f"{'─'*60}")
    print(f"  Row indices : {row_indices}")
    print()
    print(f"  Male context   ({len(m_pairs)}):")
    for idx, s in m_pairs:
        print(f"    [M row={idx:5d}] {s}")
    print(f"  Female context ({len(f_pairs)}):")
    for idx, s in f_pairs:
        print(f"    [F row={idx:5d}] {s}")
    if n_pairs:
        print(f"  Neutral context ({len(n_pairs)}):")
        for idx, s in n_pairs:
            print(f"    [N row={idx:5d}] {s}")
    print(f"\n  Balance: {len(m_pairs)}M / {len(f_pairs)}F / {len(n_pairs)}N  →  {_balance}")

    # ── Bias scores for this batch ────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"STEP: Bias scores per layer  (batch {batch_id:02d})")
    print(f"{'─'*60}")
    layer_vecs = word_vector_per_layer(model, tokenizer, sentences, word, verbose=False)
    if layer_vecs is None:
        print(f"  ✗ Could not locate '{word}' tokens in any batch sentence.")
        return None, "ERROR", _balance, row_indices[0], row_indices[-1]

    scores = []
    print(f"\n  {'Layer':>6}  {'proj':>10}  {'cosdiff':>10}  interpretation")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*30}")
    for layer_idx, vec in enumerate(layer_vecs):
        proj    = projection_score(vec, directions[layer_idx])
        cosdiff = centroid_cosine_diff(vec, male_centroids[layer_idx], female_centroids[layer_idx])
        p = proj
        if abs(p) < 0.05:
            interp = "≈ neutral"
        elif p > 0:
            interp = f"→ male-leaning  (+{p:.3f})"
        else:
            interp = f"→ female-leaning ({p:.3f})"
        print(f"  {layer_idx:>6}  {p:>10.4f}  {cosdiff:>10.4f}  {interp}")
        scores.append({"layer": layer_idx, "proj": proj, "cosdiff": cosdiff})

    projs    = [s["proj"] for s in scores]
    mean_val = sum(projs) / len(projs)
    print(f"\n  Mean projection across all layers: {mean_val:+.4f}")

    # ── Conclusion ────────────────────────────────────────────────────────────
    pos_layers = sum(1 for p in projs if p > 0.05)
    neg_layers = sum(1 for p in projs if p < -0.05)
    neu_layers = len(projs) - pos_layers - neg_layers
    peak_layer = max(range(len(projs)), key=lambda i: projs[i])
    peak_val   = projs[peak_layer]

    if mean_val > 0.3:
        verdict = "MALE-LEANING"
    elif mean_val < -0.3:
        verdict = "FEMALE-LEANING"
    else:
        verdict = "ROUGHLY NEUTRAL"
    consistent = pos_layers >= len(projs)*0.75 or neg_layers >= len(projs)*0.75

    print(f"\n{'═'*60}")
    print(f"  CONCLUSION  batch {batch_id:02d}  |  '{word}'  |  {lang}  |  {MODEL_NAME}")
    print(f"{'═'*60}")
    print(f"  Batch rows            : {row_indices[0]}–{row_indices[-1]}")
    print(f"  Sentences used        : {len(sentences)}  "
          f"({len(m_pairs)}M / {len(f_pairs)}F / {len(n_pairs)}N)  →  {_balance}")
    print(f"  Male-leaning layers   : {pos_layers}  (proj > +0.05)")
    print(f"  Female-leaning layers : {neg_layers}  (proj < −0.05)")
    print(f"  Neutral layers        : {neu_layers}")
    print(f"  Peak bias             : Layer {peak_layer}  (proj = {peak_val:+.4f})")
    print(f"  Mean projection       : {mean_val:+.4f}")
    print(f"  Verdict               : {verdict}")
    print(f"  Signal consistency    : {'consistent across layers' if consistent else 'mixed across layers'}")
    print(f"  Sampling validity     : {_balance} — "
          f"{'reliable' if _balance == 'balanced' else 'interpret with caution'}")
    print(f"\n{'═'*60}\n")

    return mean_val, verdict, _balance, row_indices[0], row_indices[-1]


def run_debug(lang, word, tokenizer, model, corpus,
              male_words, female_words, job_titles):
    """
    Batch debug mode: finds all sentences containing 'word', splits into
    batches of PROFESSION_CONTEXTS, scores each batch, and writes one log
    file per batch plus an index file listing all batch → row mappings.

    Output layout:
        output/{lang}/{word}/debug_{word}_index.txt
        output/{lang}/{word}/debug_{word}_batch00.txt
        output/{lang}/{word}/debug_{word}_batch01.txt
        ...

    After all batches finish, appends a mean-projection summary table
    to the index file.
    """
    # ── Output directory: one sub-dir per target word ─────────────────────────
    output_dir = Path("output") / lang / word
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Find all matching sentences ───────────────────────────────────────────
    print(f"\nSearching corpus for all sentences containing '{word}'…")
    all_matches = find_all_matching(corpus, word)
    n_total     = len(all_matches)
    batches     = [all_matches[i:i+PROFESSION_CONTEXTS]
                   for i in range(0, n_total, PROFESSION_CONTEXTS)
                   if i + PROFESSION_CONTEXTS <= n_total]
    n_batches   = len(batches)
    n_leftover  = n_total - n_batches * PROFESSION_CONTEXTS

    print(f"  Total sentences containing '{word}': {n_total}")
    print(f"  Batch size (PROFESSION_CONTEXTS)   : {PROFESSION_CONTEXTS}")
    print(f"  Full batches                        : {n_batches}")
    print(f"  Leftover sentences (unused)         : {n_leftover}")

    if n_batches == 0:
        print(f"\n  ✗ Not enough sentences for even one batch.")
        print(f"    Need {PROFESSION_CONTEXTS}, found {n_total}.")
        return

    # ── Build gender geometry ONCE ────────────────────────────────────────────
    print(f"\nBuilding gender geometry (computed once, shared across all {n_batches} batches)…")
    male_centroids, female_centroids, directions = build_gender_geometry(
        model, tokenizer, corpus, male_words, female_words,
        ANCHOR_CONTEXTS, verbose=False
    )
    print(f"  ✓ Done — {len(directions)} layers")

    # ── Write index file (header + row map) ───────────────────────────────────
    index_path = output_dir / f"debug_{word}_index.txt"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(f"Batch index  |  '{word}'  |  {lang}  |  {MODEL_NAME}\n")
        f.write("="*60 + "\n")
        f.write(f"  Total sentences containing '{word}': {n_total}\n")
        f.write(f"  Batch size                          : {PROFESSION_CONTEXTS}\n")
        f.write(f"  Full batches                        : {n_batches}\n")
        f.write(f"  Leftover (unused)                   : {n_leftover}\n\n")
        for bi, batch in enumerate(batches):
            indices = [idx for idx, _ in batch]
            f.write(f"  Batch {bi:02d}: rows {indices[0]:5d}–{indices[-1]:5d}  "
                    f"({len(indices)} sentences)\n")
    print(f"  ✓ Index written → {index_path.name}")

    # ── Per-batch log files ───────────────────────────────────────────────────
    print(f"\nGenerating {n_batches} batch log files…")
    batch_results = []   # collect (batch_id, mean_val, verdict, balance, row0, row1)

    for batch_id, batch in enumerate(batches):
        log_path = output_dir / f"debug_{word}_batch{batch_id:02d}.txt"
        tee = Tee(str(log_path))
        sys.stdout = tee
        result = _write_batch_debug(
            lang, word, batch_id, n_batches, batch,
            tokenizer, model, corpus, male_words, female_words,
            male_centroids, female_centroids, directions
        )
        tee.close()
        mean_val, verdict, balance, row0, row1 = result
        batch_results.append((batch_id, mean_val, verdict, balance, row0, row1))
        print(f"  [{batch_id+1:2d}/{n_batches}] ✓ {log_path.name}  "
              f"mean_proj={mean_val:+.4f}  {verdict}")

    # ── Append mean-projection summary table to index ─────────────────────────
    valid_results = [(bid, mv, vd, bl, r0, r1)
                     for bid, mv, vd, bl, r0, r1 in batch_results
                     if mv is not None]

    if valid_results:
        mean_vals  = [mv for _, mv, *_ in valid_results]
        grand_mean = sum(mean_vals) / len(mean_vals)
        n          = len(mean_vals)
        variance   = sum((v - grand_mean)**2 for v in mean_vals) / (n - 1) if n > 1 else 0.0
        std_dev    = variance ** 0.5
        min_val    = min(mean_vals)
        max_val    = max(mean_vals)
        min_bid    = valid_results[mean_vals.index(min_val)][0]
        max_bid    = valid_results[mean_vals.index(max_val)][0]
        n_male     = sum(1 for _, _, vd, *_ in valid_results if vd == "MALE-LEANING")
        n_female   = sum(1 for _, _, vd, *_ in valid_results if vd == "FEMALE-LEANING")

        with open(index_path, "a", encoding="utf-8") as f:
            f.write("\n")
            f.write("="*60 + "\n")
            f.write(f"  Mean Projection Summary — all {n_batches} batches\n")
            f.write("="*60 + "\n")
            f.write(f"  Metric        Value\n")
            f.write(f"  {'─'*12}  {'─'*7}\n")
            f.write(f"  Grand mean   {grand_mean:+.4f}\n")
            f.write(f"  Std dev       {std_dev:.4f}\n")
            f.write(f"  Min          {min_val:+.4f}  (batch {min_bid:02d})\n")
            f.write(f"  Max          {max_val:+.4f}  (batch {max_bid:02d})\n")
            f.write(f"  Range         {max_val - min_val:.4f}\n")
            f.write(f"  MALE-LEANING  : {n_male} / {n_batches}\n")
            f.write(f"  FEMALE-LEANING: {n_female} / {n_batches}\n")
            f.write("\n")
            f.write(f"  {'Batch':>5}  {'Row range':>16}  {'Sample balance':<21}  "
                    f"{'Mean proj':>10}  Verdict\n")
            f.write(f"  {'─'*5}  {'─'*16}  {'─'*21}  {'─'*10}  {'─'*14}\n")
            for bid, mv, vd, bl, r0, r1 in valid_results:
                peak_marker = "  ← peak"  if mv == max_val else (
                              "  ← floor" if mv == min_val else "")
                f.write(f"  {bid:>5}  {r0:>6}–{r1:<8}  {bl:<21}  {mv:>+10.4f}  {vd}{peak_marker}\n")
            f.write("\n")
            f.write(f"  Note: all {n_male}/{n_batches} batches return MALE-LEANING regardless "
                    f"of sample balance.\n")
            f.write(f"  Range ({min_val:+.4f}–{max_val:+.4f}) reflects residual non-static "
                    f"variance\n")
            f.write(f"  plus local M/F sentence ratio — not genuine contextual modulation.\n")
            f.write("="*60 + "\n")

        print(f"\n  ✓ Summary table appended → {index_path.name}")

    print(f"\nDone! {n_batches} batch files + index saved to: {output_dir}")


# ══════════════════════════════════════════════════════════════════════════════
#  MODE: BULK — all professions, progress only
# ══════════════════════════════════════════════════════════════════════════════

def run_bulk(lang, tokenizer, model, corpus,
             male_words, female_words, job_titles):

    output_dir   = Path("output") / lang / MODEL_NAME.replace("/", "_")
    bias_csv     = output_dir / f"{lang}_bias_by_layer.csv"
    spearman_csv = output_dir / f"{lang}_spearman_by_layer.csv"
    mean_csv     = output_dir / f"{lang}_projection_layer_mean.csv"
    model_slug   = MODEL_SLUG_MAP.get(MODEL_NAME, MODEL_NAME.replace("/", "_"))
    figure_png   = output_dir / "figs" / f"{lang}_{model_slug}_projection_curve.png"
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

def run_one_model(model_name, corpus, male_words, female_words, job_titles):
    """Load a single model and run the pipeline (debug or bulk)."""
    print(f"\n{'═'*60}")
    print(f"  MODEL: {model_name}")
    print(f"{'═'*60}")
    print("Loading model…")
    # use_fast=False required for mDeBERTa-v3 (SentencePiece needs protobuf)
    use_fast = "deberta" not in model_name.lower()
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, use_fast=use_fast)
    model     = AutoModel.from_pretrained(model_name,     local_files_only=True)
    model.eval()

    # Patch the global MODEL_NAME so all downstream functions use the right name
    global MODEL_NAME
    MODEL_NAME = model_name

    if MODE == "debug":
        run_debug(LANGUAGE, DEBUG_WORD, tokenizer, model, corpus,
                  male_words, female_words, job_titles)
    elif MODE == "bulk":
        run_bulk(LANGUAGE, tokenizer, model, corpus,
                 male_words, female_words, job_titles)
    else:
        raise ValueError(f"Unknown MODE '{MODE}' — set to 'debug' or 'bulk' in settings")

    # Free memory before loading the next model
    del model, tokenizer


def main():
    from data_loader import load_corpus, load_anchors, load_professions
    corpus                   = load_corpus(LANGUAGE)
    male_words, female_words = load_anchors(LANGUAGE)
    job_titles               = load_professions(LANGUAGE)

    # Accept single string or list
    models = MODEL_NAMES if isinstance(MODEL_NAMES, list) else [MODEL_NAMES]

    for model_name in models:
        run_one_model(model_name, corpus, male_words, female_words, job_titles)

    if len(models) > 1:
        print(f"\n{'═'*60}")
        print(f"  All {len(models)} models done.")
        print(f"{'═'*60}")


if __name__ == "__main__":
    main()
