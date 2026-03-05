"""
main.py
-------
Entry point. Edit config.py to change language, model(s), or run mode,
then run:  python src/main.py
"""

import csv
import logging
import os
from pathlib import Path

from transformers import AutoModel, AutoTokenizer

import config
from data_loader import load_anchors, load_corpus, load_professions
from debug_mode import run_debug
from embeddings import build_gender_geometry
from metrics import bias_scores_for_word, spearman_per_layer
from output import plot_curve

logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"


def run_bulk(lang, tokenizer, model, corpus, male_words, female_words,
             job_titles, model_name):
    model_slug   = config.MODEL_SLUG_MAP.get(model_name, model_name.replace("/", "_"))
    output_dir   = Path("output") / lang / model_name.replace("/", "_")
    bias_csv     = output_dir / f"{lang}_bias_by_layer.csv"
    spearman_csv = output_dir / f"{lang}_spearman_by_layer.csv"
    mean_csv     = output_dir / f"{lang}_projection_layer_mean.csv"
    figure_png   = output_dir / "figs" / f"{lang}_{model_slug}_projection_curve.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figs").mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  BULK MODE  |  language={lang}  |  {len(job_titles)} professions")
    print(f"{'='*60}\n")

    print("Building gender geometry...")
    male_centroids, female_centroids, directions = build_gender_geometry(
        model, tokenizer, corpus, male_words, female_words, config.ANCHOR_CONTEXTS
    )
    print(f"  Gender direction built across {len(directions)} layers")

    print("\nScoring professions...")
    all_records, skipped = [], []

    for idx, job in enumerate(job_titles):
        print(f"  [{idx+1:2d}/{len(job_titles)}] {job:<25}", end="  ")
        scores, found = bias_scores_for_word(
            model, tokenizer, corpus, job,
            male_centroids, female_centroids, directions,
            config.PROFESSION_CONTEXTS
        )
        if scores is None:
            skipped.append((job, found))
            print(f"SKIPPED ({found}/{config.PROFESSION_CONTEXTS} sentences)")
            continue
        for s in scores:
            all_records.append({"term": job, **s})
        mean_proj = sum(s["proj"] for s in scores) / len(scores)
        label = ("-> male" if mean_proj > 0.05 else
                 "-> female" if mean_proj < -0.05 else "approx neutral")
        print(f"OK  mean_proj={mean_proj:+.4f}  {label}")

    # Write bias CSV
    with open(bias_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["language", "model", "term", "layer", "proj", "cosdiff"])
        for r in all_records:
            writer.writerow([lang, model_name, r["term"], r["layer"], r["proj"], r["cosdiff"]])

    # Write Spearman CSV
    spearman_rows = spearman_per_layer(all_records)
    with open(spearman_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["language", "model", "layer", "spearman_rho", "spearman_p", "n"])
        for r in spearman_rows:
            writer.writerow([lang, model_name, r["layer"], r["rho"], r["p"], r["n"]])

    # Save projection curve PNG + mean CSV
    plot_curve(all_records, str(figure_png), str(mean_csv),
               f"Layer-wise mean projection ({lang} | {model_name})")

    kept = len(job_titles) - len(skipped)
    print(f"\n{'-'*60}")
    print(f"  Results saved to: {output_dir}")
    print(f"  Professions scored: {kept}/{len(job_titles)}")
    if skipped:
        print(f"  Skipped ({len(skipped)}):")
        for job, n in skipped:
            print(f"    - {job}: {n}/{config.PROFESSION_CONTEXTS} sentences")
    if spearman_rows:
        mid = spearman_rows[len(spearman_rows) // 2]
        strength = ("strong" if abs(mid["rho"]) > 0.7 else
                    "moderate" if abs(mid["rho"]) > 0.4 else "weak")
        print(f"\n  Spearman rho layer {mid['layer']} (mid-network): "
              f"{mid['rho']:.3f}  ({strength} agreement)")


def run_one_model(model_name, corpus, male_words, female_words, job_titles):
    """Load one model checkpoint and run the configured pipeline mode."""
    print(f"\n{'='*60}")
    print(f"  MODEL: {model_name}")
    print(f"{'='*60}")
    print("Loading model...")
    use_fast  = "deberta" not in model_name.lower()  # mDeBERTa needs use_fast=False
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, use_fast=use_fast)
    model     = AutoModel.from_pretrained(model_name, local_files_only=True)
    model.eval()

    if config.MODE == "debug":
        run_debug(config.LANGUAGE, config.DEBUG_WORD,
                  tokenizer, model, corpus, male_words, female_words,
                  job_titles, model_name)
    elif config.MODE == "bulk":
        run_bulk(config.LANGUAGE, tokenizer, model, corpus,
                 male_words, female_words, job_titles, model_name)
    else:
        raise ValueError(f"Unknown MODE '{config.MODE}' — set to 'debug' or 'bulk' in config.py")

    del model, tokenizer  # free GPU/RAM before loading the next model


def main():
    corpus                   = load_corpus(config.LANGUAGE)
    male_words, female_words = load_anchors(config.LANGUAGE)
    job_titles               = load_professions(config.LANGUAGE)

    models = (config.MODEL_NAMES if isinstance(config.MODEL_NAMES, list)
              else [config.MODEL_NAMES])

    for model_name in models:
        run_one_model(model_name, corpus, male_words, female_words, job_titles)

    if len(models) > 1:
        print(f"\n{'='*60}")
        print(f"  All {len(models)} models complete.")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
