"""
target_weights.py
-----------------
Inspect the model's STATIC (weight-level) representation of a target word.

Unlike contextual embeddings (which change with every sentence), the embedding
layer is a fixed lookup table — one vector per vocabulary token, with zero
context applied. These are the model's a-priori representations before any
transformer attention has processed a single word.

WHAT THIS TELLS US
------------------
If the static bias score closely matches the contextual bias scores from m_v4,
the bias is encoded directly in the model weights — not a product of the
corpus sentences or the surrounding context. This is the strongest possible
evidence that the bias is intrinsic and not context-driven.

MODES
-----
  static_bias  → project each profession's static embedding onto the static
                 gender direction; report full ranked table
                 output: output/{lang}/target_weights_bias.txt

  neighbors    → k nearest professions + anchor words to TARGET_WORD in static
                 embedding space (cosine similarity)
                 output: output/{lang}/target_weights_neighbors_{word}.txt

  subwords     → per-subword breakdown for TARGET_WORD — each subword token's
                 individual static bias score, so you can see which piece
                 drives the association
                 output: output/{lang}/target_weights_subwords_{word}.txt
"""

import os
import logging
from pathlib import Path

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

MODE        = "static_bias"   # "static_bias" | "neighbors" | "subwords"
TARGET_WORD = "ሓኪም"           # used by "neighbors" and "subwords" modes

NEIGHBORS_K = 15              # how many nearest neighbors to show

OUTPUT_DIR = Path("output") / LANGUAGE


# ══════════════════════════════════════════════════════════════════════════════
#  Core helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_embedding_matrix(model):
    """Return the full static embedding weight matrix [vocab_size, hidden_size]."""
    return model.embeddings.word_embeddings.weight  # no grad needed


def get_static_vec(model, tokenizer, word):
    """
    Look up the static embedding for each subword token of word and
    mean-pool across subwords.  Returns (vec [hidden_size], token_list).
    No forward pass — pure weight lookup.
    """
    token_ids = tokenizer.encode(word, add_special_tokens=False)
    tokens    = tokenizer.convert_ids_to_tokens(token_ids)
    W         = get_embedding_matrix(model)

    with torch.no_grad():
        vecs = W[token_ids]          # [n_subwords, hidden_size]
        vec  = vecs.mean(dim=0)      # mean-pool

    return vec.detach(), tokens


def get_static_vecs_for_list(model, tokenizer, words):
    """Return {word: (vec, tokens)} for a list of words."""
    return {w: get_static_vec(model, tokenizer, w) for w in words}


def build_static_gender_direction(model, tokenizer, male_words, female_words):
    """
    Build male centroid, female centroid, and unit gender direction entirely
    from static embeddings — no corpus, no forward pass.
    """
    def centroid(words):
        vecs = []
        for w in words:
            vec, _ = get_static_vec(model, tokenizer, w)
            vecs.append(vec)
        return torch.stack(vecs).mean(dim=0)

    male_c   = centroid(male_words)
    female_c = centroid(female_words)
    diff     = male_c - female_c
    norm     = diff.norm()
    if norm < 1e-8:
        raise ValueError("Static male and female centroids are identical.")
    direction = diff / norm
    return male_c, female_c, direction


def cosine_sim(a, b):
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))


def proj_score(vec, direction):
    return float(torch.dot(vec, direction))


def cosdiff_score(vec, male_c, female_c):
    return cosine_sim(vec, male_c) - cosine_sim(vec, female_c)


# ══════════════════════════════════════════════════════════════════════════════
#  MODE: static_bias
# ══════════════════════════════════════════════════════════════════════════════

def run_static_bias(lang, tokenizer, model, male_words, female_words, job_titles):
    """
    For every profession, compute its static embedding bias score and rank them.
    No corpus needed — purely from model weights.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "target_weights_bias.txt"

    print("\nBuilding static gender direction from embedding weights…")
    male_c, female_c, direction = build_static_gender_direction(
        model, tokenizer, male_words, female_words
    )
    sep = (male_c - female_c).norm().item()
    print(f"  ✓ Static gender direction built")
    print(f"  ‖male_centroid − female_centroid‖ = {sep:.4f}")

    print(f"\nScoring {len(job_titles)} professions from static embeddings…")
    rows = []
    for word in job_titles:
        vec, tokens = get_static_vec(model, tokenizer, word)
        proj   = proj_score(vec, direction)
        cdiff  = cosdiff_score(vec, male_c, female_c)
        n_tok  = len(tokens)
        if proj > 0.3:
            label = "MALE"
        elif proj < -0.3:
            label = "FEMALE"
        else:
            label = "NEUTRAL"
        rows.append((word, tokens, n_tok, proj, cdiff, label))

    rows.sort(key=lambda x: -x[3])   # sort by proj descending

    lines = []
    def w(line=""):
        print(line)
        lines.append(line)

    w(f"Static Embedding Bias — '{lang}'  |  {MODEL_NAME}")
    w("=" * 78)
    w(f"  Source    : model.embeddings.word_embeddings.weight  (no context, no corpus)")
    w(f"  Direction : static male centroid − static female centroid (unit vector)")
    w(f"  Anchors   : {len(male_words)}M / {len(female_words)}F")
    w(f"  ‖male − female‖ (static) : {sep:.4f}")
    w()
    w(f"  {'Word':<20}  {'Tokens':<6}  {'proj':>9}  {'cosdiff':>9}  Label")
    w(f"  {'─'*20}  {'─'*6}  {'─'*9}  {'─'*9}  {'─'*8}")
    for word, tokens, n_tok, proj, cdiff, label in rows:
        w(f"  {word:<20}  {n_tok:>6}  {proj:>+9.4f}  {cdiff:>+9.4f}  {label}")
    w()
    w("─" * 78)

    male_words_out   = [r[0] for r in rows if r[5] == "MALE"]
    female_words_out = [r[0] for r in rows if r[5] == "FEMALE"]
    neutral_words    = [r[0] for r in rows if r[5] == "NEUTRAL"]
    mean_proj_all    = sum(r[3] for r in rows) / len(rows)

    w(f"  Male-leaning  ({len(male_words_out)}): {male_words_out}")
    w(f"  Female-leaning ({len(female_words_out)}): {female_words_out}")
    w(f"  Neutral       ({len(neutral_words)}): {neutral_words}")
    w()
    w(f"  Mean proj across all professions: {mean_proj_all:+.4f}")
    w("=" * 78)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n✓ Report saved to: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MODE: neighbors
# ══════════════════════════════════════════════════════════════════════════════

def run_neighbors(lang, word, tokenizer, model, male_words, female_words, job_titles):
    """
    Find the NEIGHBORS_K nearest words (among professions + anchors) to the
    target word in static embedding space.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"target_weights_neighbors_{word}.txt"

    target_vec, target_tokens = get_static_vec(model, tokenizer, word)

    candidates = {}
    for w in job_titles:
        if w != word:
            candidates[w] = ("profession", *get_static_vec(model, tokenizer, w))
    for w in male_words:
        candidates[w] = ("male-anchor", *get_static_vec(model, tokenizer, w))
    for w in female_words:
        candidates[w] = ("female-anchor", *get_static_vec(model, tokenizer, w))

    scored = []
    for cand, (role, vec, tokens) in candidates.items():
        sim = cosine_sim(target_vec, vec)
        scored.append((cand, role, tokens, sim))
    scored.sort(key=lambda x: -x[3])

    lines = []
    def w(line=""):
        print(line)
        lines.append(line)

    w(f"Static Embedding Neighbors — '{word}'  |  {lang}  |  {MODEL_NAME}")
    w("=" * 70)
    w(f"  Target word  : '{word}'  →  {target_tokens}")
    w(f"  Space        : static embedding layer (weight lookup, no context)")
    w(f"  Candidates   : {len(job_titles)-1} professions + {len(male_words)} M anchors + {len(female_words)} F anchors")
    w(f"  Showing top  : {NEIGHBORS_K}")
    w()
    w(f"  {'Rank':>4}  {'Word':<22}  {'Role':<14}  {'Tokens':<6}  {'cos_sim':>8}")
    w(f"  {'─'*4}  {'─'*22}  {'─'*14}  {'─'*6}  {'─'*8}")
    for rank, (cand, role, tokens, sim) in enumerate(scored[:NEIGHBORS_K], 1):
        w(f"  {rank:>4}  {cand:<22}  {role:<14}  {len(tokens):>6}  {sim:>8.4f}")
    w()
    w("─" * 70)
    anchor_hits = [(c, r, s) for c, r, _, s in scored[:NEIGHBORS_K]
                   if "anchor" in r]
    if anchor_hits:
        w(f"  Gender anchors in top-{NEIGHBORS_K}:")
        for c, r, s in anchor_hits:
            w(f"    {r:<16}  '{c}'  sim={s:.4f}")
    else:
        w(f"  No gender anchors in top-{NEIGHBORS_K} — target is far from anchor cluster.")
    w("=" * 70)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n✓ Report saved to: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MODE: subwords
# ══════════════════════════════════════════════════════════════════════════════

def run_subwords(lang, word, tokenizer, model, male_words, female_words):
    """
    Show each subword token's individual static embedding bias score.
    Identifies WHICH subword piece drives the gender association.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"target_weights_subwords_{word}.txt"

    male_c, female_c, direction = build_static_gender_direction(
        model, tokenizer, male_words, female_words
    )

    token_ids = tokenizer.encode(word, add_special_tokens=False)
    tokens    = tokenizer.convert_ids_to_tokens(token_ids)
    W         = get_embedding_matrix(model)

    lines = []
    def w(line=""):
        print(line)
        lines.append(line)

    w(f"Subword Static Bias Breakdown — '{word}'  |  {lang}  |  {MODEL_NAME}")
    w("=" * 68)
    w(f"  Word       : '{word}'")
    w(f"  Subwords   : {tokens}")
    w(f"  Token IDs  : {token_ids}")
    w(f"  (Each subword has its own static vector — we project each individually)")
    w()
    w(f"  {'Subword':<16}  {'token_id':>9}  {'proj':>9}  {'cosdiff':>9}  {'norm':>8}  Driver?")
    w(f"  {'─'*16}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*8}  {'─'*7}")

    subword_projs = []
    with torch.no_grad():
        for tid, tok in zip(token_ids, tokens):
            vec  = W[tid].detach()
            proj = proj_score(vec, direction)
            cdif = cosdiff_score(vec, male_c, female_c)
            nrm  = float(vec.norm())
            subword_projs.append(proj)
            w(f"  {tok:<16}  {tid:>9}  {proj:>+9.4f}  {cdif:>+9.4f}  {nrm:>8.4f}")

    mean_vec, _ = get_static_vec(model, tokenizer, word)
    mean_proj   = proj_score(mean_vec, direction)
    mean_cdiff  = cosdiff_score(mean_vec, male_c, female_c)

    max_driver = tokens[subword_projs.index(max(subword_projs, key=abs))]

    w()
    w("─" * 68)
    w(f"  Mean-pooled vector   proj = {mean_proj:>+.4f}   cosdiff = {mean_cdiff:>+.4f}")
    w(f"  Strongest driver     : '{max_driver}'  "
      f"(proj = {max(subword_projs, key=abs):>+.4f})")
    w()

    if max(subword_projs, key=abs) / (mean_proj + 1e-8) > 1.5:
        w("  → One subword token dominates the bias signal.")
        w("    The mean-pooled score is pulled by a single piece.")
    else:
        w("  → Bias is distributed across subword tokens.")
        w("    No single piece dominates — the full word drives the association.")
    w("=" * 68)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n✓ Report saved to: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    from data_loader import load_anchors, load_professions

    male_words, female_words = load_anchors(LANGUAGE)
    job_titles               = load_professions(LANGUAGE)

    print("Loading model…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model     = AutoModel.from_pretrained(MODEL_NAME,     local_files_only=True)
    model.eval()

    if MODE == "static_bias":
        run_static_bias(LANGUAGE, tokenizer, model, male_words, female_words, job_titles)
    elif MODE == "neighbors":
        run_neighbors(LANGUAGE, TARGET_WORD, tokenizer, model,
                      male_words, female_words, job_titles)
    elif MODE == "subwords":
        run_subwords(LANGUAGE, TARGET_WORD, tokenizer, model,
                     male_words, female_words)
    else:
        raise ValueError(f"Unknown MODE '{MODE}' — set to 'static_bias', 'neighbors', or 'subwords'")


if __name__ == "__main__":
    main()
