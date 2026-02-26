"""
check_rep.py
------------
For each sentence a target word appears in, extract its contextual vector
(one per transformer layer) and measure how unique / diverse those
representations are across sentences.

WHY THIS MATTERS
----------------
The bias score for a profession word is computed from the AVERAGE of its
vectors across PROFESSION_CONTEXTS sentences. If all those vectors are
nearly identical (cosine similarity ≈ 1.0), the model is ignoring context
and the averaging adds no value. If they are diverse (cosine similarity
much lower than 1.0), the model is encoding context richly and averaging
gives a stable, representative embedding.

METRICS (per layer)
-------------------
  mean_cos     : mean pairwise cosine similarity across all sentence vectors
                 (lower = more unique / context-sensitive)
  std_cos      : spread of pairwise similarities
  min_cos      : most dissimilar pair
  max_cos      : most similar pair
  n_near_dups  : number of pairs with cosine > 0.99 (true near-duplicates)
  norm_mean    : mean L2 norm of raw (un-normalized) vectors
  norm_std     : std of L2 norms (spread in magnitude)

MODES
-----
  single → full per-layer report for one target word
           output: output/{lang}/check_rep_{word}.txt
  bulk   → one summary row per profession word
           output: output/{lang}/check_rep_bulk.csv
"""

import os
import csv
import logging
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"


# ══════════════════════════════════════════════════════════════════════════════
#  Settings
# ══════════════════════════════════════════════════════════════════════════════

LANGUAGE   = "ti"
MODEL_NAME = "xlm-roberta-base"

MODE        = "single"   # "single" | "bulk"
TARGET_WORD = "ሓኪም"      # used when MODE = "single"

# Maximum sentences to use per word.
# single: None = use every matching sentence in the corpus (full picture).
# bulk:   keep lower — 30 professions × many sentences can be slow on CPU.
MAX_SENTS_SINGLE = None   # all available
MAX_SENTS_BULK   = 50     # per profession

OUTPUT_DIR = Path("output") / LANGUAGE


# ══════════════════════════════════════════════════════════════════════════════
#  Core helpers
# ══════════════════════════════════════════════════════════════════════════════

def find_all_matching(corpus, word):
    """Return (row_idx, sentence) for every corpus sentence containing word."""
    return [(idx, s) for idx, s in enumerate(corpus) if word in s]


def extract_all_vectors(model, tokenizer, sentences, word):
    """
    Run forward passes for every sentence, extract the hidden states for
    the target word's subword tokens (mean-pooled), and return them organised
    by sentence.

    Returns
    -------
    all_layer_vecs : list[list[Tensor]]
        all_layer_vecs[sent_idx][layer_idx] = 1D tensor (hidden_size,)
    word_tokens    : list[str]   subword tokens the word was split into
    n_skipped      : int         sentences where word could not be located
    """
    word_tokens = tokenizer.tokenize(word)
    all_layer_vecs = []
    n_skipped = 0

    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt",
                           truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        sent_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        pos = None
        for i in range(len(sent_tokens) - len(word_tokens) + 1):
            if sent_tokens[i:i + len(word_tokens)] == word_tokens:
                pos = (i, i + len(word_tokens))
                break

        if pos is None:
            n_skipped += 1
            continue

        start, end = pos
        layer_vecs = [
            hs[0, start:end].mean(dim=0)
            for hs in outputs.hidden_states
        ]
        all_layer_vecs.append(layer_vecs)

    return all_layer_vecs, word_tokens, n_skipped


def uniqueness_stats(layer_vecs):
    """
    Compute pairwise cosine similarity stats for a list of 1D tensors
    (one per sentence, all for the same layer).
    Returns None if fewer than 2 vectors.
    """
    if len(layer_vecs) < 2:
        return None

    mat    = torch.stack(layer_vecs)       # [N, hidden_size]
    normed = F.normalize(mat, dim=1)       # unit-norm rows
    sim    = (normed @ normed.T).cpu()     # [N, N]

    n = sim.shape[0]
    i_idx, j_idx = torch.triu_indices(n, n, offset=1)
    upper = sim[i_idx, j_idx]             # upper triangle = unique pairs

    norms = mat.norm(dim=1).cpu()

    return {
        "n":           n,
        "n_pairs":     len(upper),
        "mean_cos":    float(upper.mean()),
        "std_cos":     float(upper.std()),
        "min_cos":     float(upper.min()),
        "max_cos":     float(upper.max()),
        "n_near_dups": int((upper > 0.99).sum()),
        "norm_mean":   float(norms.mean()),
        "norm_std":    float(norms.std()),
    }


def _layer_to_vecs(all_layer_vecs, n_layers):
    """Transpose all_layer_vecs[sent][layer] → dict layer → list of tensors."""
    d = defaultdict(list)
    for sent_vecs in all_layer_vecs:
        for l, vec in enumerate(sent_vecs):
            d[l].append(vec)
    return d


# ══════════════════════════════════════════════════════════════════════════════
#  Single mode
# ══════════════════════════════════════════════════════════════════════════════

def run_single(lang, word, tokenizer, model, corpus):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_DIR / f"check_rep_{word}.txt"

    all_matches = find_all_matching(corpus, word)
    n_total     = len(all_matches)

    if MAX_SENTS_SINGLE is not None:
        all_matches = all_matches[:MAX_SENTS_SINGLE]

    sentences = [s for _, s in all_matches]
    row_idxs  = [idx for idx, _ in all_matches]

    print(f"\nWord: '{word}'  |  lang={lang}  |  model={MODEL_NAME}")
    print(f"  Corpus size                   : {len(corpus)}")
    print(f"  Sentences containing '{word}' : {n_total}")
    if MAX_SENTS_SINGLE is not None:
        print(f"  Capped at MAX_SENTS_SINGLE    : {MAX_SENTS_SINGLE}")
    print(f"  Using                          : {len(sentences)} sentences")
    print(f"\n  Extracting vectors (this may take a moment)…")

    all_layer_vecs, word_tokens, n_skipped = extract_all_vectors(
        model, tokenizer, sentences, word
    )
    n_found   = len(all_layer_vecs)
    n_layers  = len(all_layer_vecs[0]) if n_found > 0 else 0
    l2v       = _layer_to_vecs(all_layer_vecs, n_layers)

    print(f"  ✓ Vectors extracted from {n_found} sentences  "
          f"({n_skipped} skipped — word not found by tokenizer)")

    if n_found < 2:
        print("  ✗ Need at least 2 sentences to compute pairwise stats.")
        return

    print(f"  Computing pairwise cosine similarity across {n_found} vectors "
          f"per layer ({n_found*(n_found-1)//2} pairs)…")
    stats_per_layer = {l: uniqueness_stats(l2v[l]) for l in range(n_layers)}

    # ── Write report ──────────────────────────────────────────────────────────
    lines = []

    def w(line=""):
        print(line)
        lines.append(line)

    w(f"Representation Uniqueness Check")
    w("=" * 72)
    w(f"  Word         : '{word}'")
    w(f"  Language     : {lang}")
    w(f"  Model        : {MODEL_NAME}")
    w(f"  Corpus size  : {len(corpus)} sentences")
    w(f"  Total hits   : {n_total} sentences containing '{word}'")
    w(f"  Used         : {n_found} (after tokenizer filter, {n_skipped} skipped)")
    w(f"  Row indices  : {row_idxs[0]}–{row_idxs[-1]}  "
      f"({'all' if MAX_SENTS_SINGLE is None else f'first {MAX_SENTS_SINGLE}'})")
    w(f"  Tokenization : {word_tokens}  ({len(word_tokens)} subword token(s), mean-pooled)")
    w()
    w("  Interpretation guide:")
    w("    mean_cos ≈ 1.0 → all representations nearly identical  (model ignores context)")
    w("    mean_cos ≈ 0.7 → moderate diversity  (some context sensitivity)")
    w("    mean_cos ≈ 0.3 → high diversity  (strongly context-sensitive)")
    w("    n_near_dups    → pairs with cosine > 0.99  (true near-duplicates)")
    w()
    w("─" * 72)
    w(f"  {'Layer':>5}  {'n':>5}  {'mean_cos':>9}  {'std_cos':>8}  "
      f"{'min_cos':>8}  {'max_cos':>8}  {'n_dups':>7}  "
      f"{'norm_mean':>9}  {'norm_std':>8}")
    w(f"  {'─'*5}  {'─'*5}  {'─'*9}  {'─'*8}  "
      f"{'─'*8}  {'─'*8}  {'─'*7}  {'─'*9}  {'─'*8}")

    for l in range(n_layers):
        st = stats_per_layer[l]
        if st is None:
            continue
        w(f"  {l:>5}  {st['n']:>5}  {st['mean_cos']:>9.4f}  {st['std_cos']:>8.4f}  "
          f"{st['min_cos']:>8.4f}  {st['max_cos']:>8.4f}  {st['n_near_dups']:>7}  "
          f"{st['norm_mean']:>9.4f}  {st['norm_std']:>8.4f}")

    # ── Summary ────────────────────────────────────────────────────────────────
    valid = [(l, s) for l, s in stats_per_layer.items() if s is not None]
    most_diverse     = min(valid, key=lambda x: x[1]["mean_cos"])
    most_uniform     = max(valid, key=lambda x: x[1]["mean_cos"])
    most_dups        = max(valid, key=lambda x: x[1]["n_near_dups"])
    mean_cos_overall = sum(s["mean_cos"] for _, s in valid) / len(valid)
    total_dups       = sum(s["n_near_dups"] for _, s in valid)

    w()
    w("─" * 72)
    w("  Summary:")
    w(f"    Most diverse layer     : {most_diverse[0]}  "
      f"(mean_cos = {most_diverse[1]['mean_cos']:.4f})")
    w(f"    Most uniform layer     : {most_uniform[0]}  "
      f"(mean_cos = {most_uniform[1]['mean_cos']:.4f})")
    w(f"    Most near-duplicates   : Layer {most_dups[0]}  "
      f"({most_dups[1]['n_near_dups']} pairs > 0.99)")
    w()
    w(f"  Mean cosine similarity across all layers : {mean_cos_overall:.4f}")
    w(f"  Total near-duplicate pairs (all layers)  : {total_dups}")
    w()

    if mean_cos_overall > 0.95:
        verdict = "WARNING — representations nearly identical. Model ignores context."
    elif mean_cos_overall > 0.85:
        verdict = "CAUTION  — moderate variation. Representations somewhat similar."
    elif mean_cos_overall > 0.6:
        verdict = "OK       — meaningful contextual variation across sentences."
    else:
        verdict = "GOOD     — highly diverse representations. Strong context sensitivity."

    w(f"  Verdict: {verdict}")
    w("=" * 72)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n✓ Report saved to: {log_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Bulk mode
# ══════════════════════════════════════════════════════════════════════════════

def run_bulk(lang, tokenizer, model, corpus, job_titles):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUT_DIR / "check_rep_bulk.csv"

    print(f"\nBULK MODE  |  lang={lang}  |  {len(job_titles)} professions  "
          f"|  max {MAX_SENTS_BULK} sentences each")
    print("─" * 60)

    rows = []
    for j_idx, word in enumerate(job_titles):
        all_matches = find_all_matching(corpus, word)
        n_total     = len(all_matches)
        sample      = all_matches[:MAX_SENTS_BULK]
        sentences   = [s for _, s in sample]

        print(f"  [{j_idx+1:2d}/{len(job_titles)}] {word:<22}  "
              f"({n_total} hits, using {len(sentences)})", end="  ")

        if len(sentences) < 2:
            print("SKIPPED (< 2 sentences)")
            continue

        all_layer_vecs, word_tokens, n_skipped = extract_all_vectors(
            model, tokenizer, sentences, word
        )
        n_found  = len(all_layer_vecs)
        n_layers = len(all_layer_vecs[0]) if n_found > 0 else 0

        if n_found < 2:
            print(f"SKIPPED (only {n_found} valid after tokenizer)")
            continue

        l2v  = _layer_to_vecs(all_layer_vecs, n_layers)
        all_stats    = [uniqueness_stats(l2v[l]) for l in range(n_layers)]
        valid        = [s for s in all_stats if s is not None]
        mean_cos_avg = sum(s["mean_cos"] for s in valid) / len(valid)
        total_dups   = sum(s["n_near_dups"] for s in valid)

        # Mid-network layer as representative point
        mid_l  = n_layers // 2
        mid_st = all_stats[mid_l]

        rows.append({
            "language":              lang,
            "model":                 MODEL_NAME,
            "word":                  word,
            "n_subword_tokens":      len(word_tokens),
            "n_corpus_hits":         n_total,
            "n_used":                n_found,
            "n_skipped":             n_skipped,
            "n_layers":              n_layers,
            "mean_cos_all_layers":   round(mean_cos_avg, 6),
            "total_near_dups":       total_dups,
            f"mean_cos_layer{mid_l}": round(mid_st["mean_cos"], 6) if mid_st else "",
            f"n_dups_layer{mid_l}":  mid_st["n_near_dups"] if mid_st else "",
        })

        if mean_cos_avg > 0.95:
            tag = "⚠ very uniform"
        elif mean_cos_avg > 0.85:
            tag = "~ moderate"
        else:
            tag = "✓ diverse"
        print(f"mean_cos={mean_cos_avg:.4f}  {tag}")

    if rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n✓ Bulk results saved to: {out_csv}")
    else:
        print("\n✗ No results to write.")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    from data_loader import load_corpus, load_professions

    corpus     = load_corpus(LANGUAGE)
    job_titles = load_professions(LANGUAGE)

    print("Loading model…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model     = AutoModel.from_pretrained(MODEL_NAME,     local_files_only=True)
    model.eval()

    if MODE == "single":
        run_single(LANGUAGE, TARGET_WORD, tokenizer, model, corpus)
    elif MODE == "bulk":
        run_bulk(LANGUAGE, tokenizer, model, corpus, job_titles)
    else:
        raise ValueError(f"Unknown MODE '{MODE}' — set to 'single' or 'bulk'")


if __name__ == "__main__":
    main()
