"""
debug_mode.py
-------------
Verbose single-profession trace mode.

Splits all corpus sentences for the target word into batches of
PROFESSION_CONTEXTS, scores each batch independently, and writes one
log file per batch plus an index file with a summary table.

Output layout:
    output/{lang}/{word}/debug_{word}_index.txt
    output/{lang}/{word}/debug_{word}_batch00.txt
    output/{lang}/{word}/debug_{word}_batch01.txt
    ...
"""

import sys
from pathlib import Path

import config
from data_loader import find_all_matching, classify_sentence
from embeddings import build_gender_geometry, word_vector_per_layer
from metrics import projection_score, centroid_cosine_diff
from output import Tee


def _write_batch_debug(lang, word, batch_id, n_batches, batch_with_indices,
                        tokenizer, model, corpus,
                        male_words, female_words,
                        male_centroids, female_centroids, directions,
                        model_name):
    """
    Write the full debug log for one batch of profession sentences.
    stdout must already be redirected to a Tee before calling this.

    Returns (mean_proj, verdict, balance_label, row_start, row_end).
    """
    sentences   = [s   for _, s in batch_with_indices]
    row_indices = [idx for idx, _ in batch_with_indices]

    # Header
    print("\n" + "="*60)
    print(f"  DEBUG BATCH {batch_id:02d} / {n_batches-1}  |  lang={lang}  |  word='{word}'")
    print("="*60)

    # Corpus overview
    n_hits = sum(1 for s in corpus if word in s)
    print(f"\n{'-'*60}")
    print("STEP: Corpus overview")
    print(f"{'-'*60}")
    print(f"  Total corpus sentences         : {len(corpus)}")
    print(f"  Sentences containing '{word}'  : {n_hits}")
    print(f"  Full batches of {config.PROFESSION_CONTEXTS}"
          f"             : {n_hits // config.PROFESSION_CONTEXTS}  "
          f"({n_hits % config.PROFESSION_CONTEXTS} leftover, unused)")
    print(f"  This batch                     : rows {row_indices[0]}-{row_indices[-1]}")

    # Anchor overview
    print(f"\n{'-'*60}")
    print("STEP: Anchor word overview")
    print(f"{'-'*60}")
    print(f"  Male anchors   ({len(male_words)}): {male_words[:6]}{'...' if len(male_words)>6 else ''}")
    print(f"  Female anchors ({len(female_words)}): {female_words[:6]}{'...' if len(female_words)>6 else ''}")
    print(f"  ANCHOR_CONTEXTS = {config.ANCHOR_CONTEXTS}")
    print(f"\n  Anchor coverage (need {config.ANCHOR_CONTEXTS} sentences each):")
    print(f"  {'Gender':<8}  {'Anchor':<16}  {'Available':>10}  Status")
    for w in male_words:
        avail  = sum(1 for s in corpus if w in s)
        status = "OK" if avail >= config.ANCHOR_CONTEXTS else "INSUFFICIENT"
        print(f"  {'M':<8}  {w:<16}  {avail:>10}  {status}")
    for w in female_words:
        avail  = sum(1 for s in corpus if w in s)
        status = "OK" if avail >= config.ANCHOR_CONTEXTS else "INSUFFICIENT"
        print(f"  {'F':<8}  {w:<16}  {avail:>10}  {status}")

    # Gender geometry summary
    sample = [0, 4, 8, 12] if len(directions) > 12 else list(range(len(directions)))
    print(f"\n{'-'*60}")
    print("STEP: Gender geometry  (pre-computed, shared across all batches)")
    print(f"{'-'*60}")
    print(f"  Gender direction for {len(directions)} layers")
    print("  Centroid separation ||male - female|| at sample layers:")
    for l in sample:
        raw_norm = (male_centroids[l] - female_centroids[l]).norm()
        print(f"    Layer {l:2d}: {raw_norm:.4f}")

    # Context sentences for this batch
    m_pairs = [(idx, s) for idx, s in batch_with_indices if classify_sentence(s) == "M"]
    f_pairs = [(idx, s) for idx, s in batch_with_indices if classify_sentence(s) == "F"]
    n_pairs = [(idx, s) for idx, s in batch_with_indices if classify_sentence(s) == "N"]
    _diff   = abs(len(m_pairs) - len(f_pairs))
    _balance = ("balanced"             if _diff <= 2 else
                "male-biased sample"   if len(m_pairs) > len(f_pairs) else
                "female-biased sample")

    print(f"\n{'-'*60}")
    print(f"STEP: Batch {batch_id:02d} context sentences  (row indices shown)")
    print(f"{'-'*60}")
    print(f"  Row indices : {row_indices}\n")
    print(f"  Male context ({len(m_pairs)}):")
    for idx, s in m_pairs:
        print(f"    [M row={idx:5d}] {s}")
    print(f"  Female context ({len(f_pairs)}):")
    for idx, s in f_pairs:
        print(f"    [F row={idx:5d}] {s}")
    if n_pairs:
        print(f"  Neutral context ({len(n_pairs)}):")
        for idx, s in n_pairs:
            print(f"    [N row={idx:5d}] {s}")
    print(f"\n  Balance: {len(m_pairs)}M / {len(f_pairs)}F / {len(n_pairs)}N  ->  {_balance}")

    # Bias scores for this batch
    print(f"\n{'-'*60}")
    print(f"STEP: Bias scores per layer  (batch {batch_id:02d})")
    print(f"{'-'*60}")
    layer_vecs = word_vector_per_layer(model, tokenizer, sentences, word, verbose=False)
    if layer_vecs is None:
        print(f"  Could not locate '{word}' tokens in any batch sentence.")
        return None, "ERROR", _balance, row_indices[0], row_indices[-1]

    scores = []
    print(f"\n  {'Layer':>6}  {'proj':>10}  {'cosdiff':>10}  interpretation")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*30}")
    for i, vec in enumerate(layer_vecs):
        proj    = projection_score(vec, directions[i])
        cosdiff = centroid_cosine_diff(vec, male_centroids[i], female_centroids[i])
        interp  = ("approx neutral"      if abs(proj) < 0.05 else
                   f"male-leaning  (+{proj:.3f})" if proj > 0 else
                   f"female-leaning ({proj:.3f})")
        print(f"  {i:>6}  {proj:>10.4f}  {cosdiff:>10.4f}  {interp}")
        scores.append({"layer": i, "proj": proj, "cosdiff": cosdiff})

    projs    = [s["proj"] for s in scores]
    mean_val = sum(projs) / len(projs)
    print(f"\n  Mean projection across all layers: {mean_val:+.4f}")

    # Conclusion
    pos_layers = sum(1 for p in projs if p >  0.05)
    neg_layers = sum(1 for p in projs if p < -0.05)
    neu_layers = len(projs) - pos_layers - neg_layers
    peak_layer = max(range(len(projs)), key=lambda i: projs[i])
    peak_val   = projs[peak_layer]

    verdict    = ("MALE-LEANING"    if mean_val >  0.3 else
                  "FEMALE-LEANING"  if mean_val < -0.3 else
                  "ROUGHLY NEUTRAL")
    consistent = (pos_layers >= len(projs) * 0.75 or
                  neg_layers >= len(projs) * 0.75)

    print(f"\n{'='*60}")
    print(f"  CONCLUSION  batch {batch_id:02d}  |  '{word}'  |  {lang}  |  {model_name}")
    print(f"{'='*60}")
    print(f"  Batch rows            : {row_indices[0]}-{row_indices[-1]}")
    print(f"  Sentences used        : {len(sentences)}  "
          f"({len(m_pairs)}M / {len(f_pairs)}F / {len(n_pairs)}N)  ->  {_balance}")
    print(f"  Male-leaning layers   : {pos_layers}  (proj > +0.05)")
    print(f"  Female-leaning layers : {neg_layers}  (proj < -0.05)")
    print(f"  Neutral layers        : {neu_layers}")
    print(f"  Peak bias             : Layer {peak_layer}  (proj = {peak_val:+.4f})")
    print(f"  Mean projection       : {mean_val:+.4f}")
    print(f"  Verdict               : {verdict}")
    print(f"  Signal consistency    : {'consistent' if consistent else 'mixed'} across layers")
    print(f"  Sampling validity     : {_balance}")
    print(f"\n{'='*60}\n")

    return mean_val, verdict, _balance, row_indices[0], row_indices[-1]


def run_debug(lang, word, tokenizer, model, corpus,
              male_words, female_words, job_titles, model_name):
    """
    Batch debug mode: score every full batch of PROFESSION_CONTEXTS sentences
    containing `word`, writing one log file per batch and an index summary.
    """
    output_dir = Path(config.OUTPUT_ROOT) / lang / word
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSearching corpus for all sentences containing '{word}'...")
    all_matches = find_all_matching(corpus, word)
    n_total     = len(all_matches)
    batches     = [all_matches[i : i + config.PROFESSION_CONTEXTS]
                   for i in range(0, n_total, config.PROFESSION_CONTEXTS)
                   if i + config.PROFESSION_CONTEXTS <= n_total]
    n_batches   = len(batches)
    n_leftover  = n_total - n_batches * config.PROFESSION_CONTEXTS

    print(f"  Total matches  : {n_total}")
    print(f"  Batch size     : {config.PROFESSION_CONTEXTS}")
    print(f"  Full batches   : {n_batches}")
    print(f"  Leftover (unused): {n_leftover}")

    if n_batches == 0:
        print(f"\n  Not enough sentences for one batch "
              f"(need {config.PROFESSION_CONTEXTS}, found {n_total}).")
        return

    # Build gender geometry once — shared across all batches
    print(f"\nBuilding gender geometry (shared across {n_batches} batches)...")
    male_centroids, female_centroids, directions = build_gender_geometry(
        model, tokenizer, corpus, male_words, female_words,
        config.ANCHOR_CONTEXTS, verbose=False
    )
    print(f"  Done — {len(directions)} layers")

    # Write index file (header + batch row map)
    index_path = output_dir / f"debug_{word}_index.txt"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(f"Batch index  |  '{word}'  |  {lang}  |  {model_name}\n")
        f.write("="*60 + "\n")
        f.write(f"  Total matches: {n_total}\n")
        f.write(f"  Batch size   : {config.PROFESSION_CONTEXTS}\n")
        f.write(f"  Full batches : {n_batches}\n")
        f.write(f"  Leftover     : {n_leftover}\n\n")
        for bi, batch in enumerate(batches):
            indices = [idx for idx, _ in batch]
            f.write(f"  Batch {bi:02d}: rows {indices[0]:5d}-{indices[-1]:5d}  "
                    f"({len(indices)} sentences)\n")
    print(f"  Index written -> {index_path.name}")

    # Per-batch log files
    print(f"\nGenerating {n_batches} batch log files...")
    batch_results = []

    for batch_id, batch in enumerate(batches):
        log_path = output_dir / f"debug_{word}_batch{batch_id:02d}.txt"
        tee = Tee(str(log_path))
        sys.stdout = tee
        result = _write_batch_debug(
            lang, word, batch_id, n_batches, batch,
            tokenizer, model, corpus, male_words, female_words,
            male_centroids, female_centroids, directions, model_name
        )
        tee.close()
        mean_val, verdict, balance, row0, row1 = result
        batch_results.append((batch_id, mean_val, verdict, balance, row0, row1))
        print(f"  [{batch_id+1:2d}/{n_batches}] {log_path.name}  "
              f"mean_proj={mean_val:+.4f}  {verdict}")

    # Append summary table to index
    valid = [(bid, mv, vd, bl, r0, r1)
             for bid, mv, vd, bl, r0, r1 in batch_results if mv is not None]

    if valid:
        mean_vals  = [mv for _, mv, *_ in valid]
        grand_mean = sum(mean_vals) / len(mean_vals)
        n          = len(mean_vals)
        variance   = sum((v - grand_mean)**2 for v in mean_vals) / (n - 1) if n > 1 else 0.0
        std_dev    = variance ** 0.5
        min_val    = min(mean_vals)
        max_val    = max(mean_vals)
        min_bid    = valid[mean_vals.index(min_val)][0]
        max_bid    = valid[mean_vals.index(max_val)][0]
        n_male     = sum(1 for _, _, vd, *_ in valid if vd == "MALE-LEANING")
        n_female   = sum(1 for _, _, vd, *_ in valid if vd == "FEMALE-LEANING")

        with open(index_path, "a", encoding="utf-8") as f:
            f.write("\n" + "="*60 + "\n")
            f.write(f"  Mean Projection Summary — all {n_batches} batches\n")
            f.write("="*60 + "\n")
            f.write(f"  Grand mean   {grand_mean:+.4f}\n")
            f.write(f"  Std dev       {std_dev:.4f}\n")
            f.write(f"  Min          {min_val:+.4f}  (batch {min_bid:02d})\n")
            f.write(f"  Max          {max_val:+.4f}  (batch {max_bid:02d})\n")
            f.write(f"  MALE-LEANING  : {n_male}/{n_batches}\n")
            f.write(f"  FEMALE-LEANING: {n_female}/{n_batches}\n\n")
            f.write(f"  {'Batch':>5}  {'Row range':>16}  {'Balance':<21}  "
                    f"{'Mean proj':>10}  Verdict\n")
            f.write(f"  {'─'*5}  {'─'*16}  {'─'*21}  {'─'*10}  {'─'*14}\n")
            for bid, mv, vd, bl, r0, r1 in valid:
                marker = ("  <- peak"  if mv == max_val else
                          "  <- floor" if mv == min_val else "")
                f.write(f"  {bid:>5}  {r0:>6}-{r1:<8}  {bl:<21}  {mv:>+10.4f}  {vd}{marker}\n")
            f.write("\n" + "="*60 + "\n")

        print(f"\n  Summary table appended -> {index_path.name}")

    print(f"\nDone! {n_batches} batch files + index -> {output_dir}")
