"""
verify_citations.py
Extracts text from each citation PDF and searches for key phrases
that confirm (or contradict) each claim made in the Related Work section.
Outputs a structured verification report.
"""

import pdfplumber
import os
import re

CITATIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "citations")

# Each entry: (citation_key, claim_summary, search_terms)
# search_terms: list of strings to look for (any match counts as a hit)
CLAIMS = [
    (
        "bolukbasi2016man",
        "gender direction from anchor sets reveals profession-gender associations; projection-based debiasing",
        ["gender direction", "hard debiasing", "gender subspace", "profession", "anchor", "debiasing"],
    ),
    (
        "caliskan2017semantics",
        "introduced WEAT — statistical framework for measuring implicit associations",
        ["Word Embedding Association Test", "WEAT", "implicit association", "statistical framework"],
    ),
    (
        "garg2018word",
        "embedding bias correlates with historical societal trends",
        ["historical", "societal", "over time", "100 years", "census", "correlation"],
    ),
    (
        "gonen2019lipstick",
        "removing a single bias direction does not eliminate deeper geometric bias",
        ["lipstick", "removing", "does not remove", "clustering", "geometric", "remaining bias", "not removed"],
    ),
    (
        "bommasani2020reductions",
        "contextual embeddings aggregated across occurrences → stable word-level vectors",
        ["reduction", "static embedding", "aggregate", "occurrences", "word-level", "contextual-to-static"],
    ),
    (
        "rogers2020primer",
        "surveys uneven distribution of linguistic information across transformer layers",
        ["layer", "linguistic", "distribution", "different layers", "uneven", "BERTology", "what BERT"],
    ),
    (
        "zhao2019gender",
        "contextualised BERT representations preserve gender bias structure of static embeddings",
        ["gender bias", "BERT", "contextualized", "ELMo", "coreference", "preserve", "static"],
    ),
    (
        "zhao-etal-2020-gender",
        "bias varies across languages; debiasing in one language does not transfer to others",
        ["multilingual", "cross-lingual", "debiasing", "transfer", "languages", "does not", "generaliz"],
    ),
    (
        "guo2021detecting",
        "contextualised embeddings encode intersectional associations, not merely single-axis gender bias",
        ["intersectional", "intersectionality", "distribution", "multiple", "race", "gender"],
    ),
    (
        "lauscher2020zero",
        "zero-shot cross-lingual transfer degrades for typologically distant and low-resource languages",
        ["zero-shot", "low-resource", "distant", "degrad", "typolog", "transfer", "limitation"],
    ),
    (
        "he2021deberta",
        "introduced Disentangled Attention separating content and positional information",
        ["disentangled attention", "disentangled", "content", "position", "separate", "two vectors"],
    ),
    (
        "he2023debertav3",
        "extended to multilingual pretraining via ELECTRA-style objective; produces mdeberta-v3-base",
        ["ELECTRA", "multilingual", "mdeberta", "v3", "replaced token detection", "gradient-disentangled"],
    ),
    (
        "liang2023xlmv",
        "enlarged SentencePiece vocabulary from 250K (XLM-R) to 1M tokens (XLM-V) for low-resource coverage",
        ["1M", "1 million", "vocabulary", "250", "XLM-R", "low-resource", "tokenization", "SentencePiece"],
    ),
]


def extract_text(pdf_path, max_pages=12):
    """Extract text from first max_pages pages of a PDF."""
    text_pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages[:max_pages]):
                t = page.extract_text()
                if t:
                    text_pages.append((i + 1, t))
    except Exception as e:
        return [], str(e)
    return text_pages, None


def find_hits(text_pages, search_terms):
    """Return list of (page, snippet) for each matching term."""
    hits = []
    seen_terms = set()
    for term in search_terms:
        pat = re.compile(re.escape(term), re.IGNORECASE)
        for page_num, page_text in text_pages:
            for m in pat.finditer(page_text):
                if term.lower() in seen_terms:
                    break
                seen_terms.add(term.lower())
                start = max(0, m.start() - 120)
                end = min(len(page_text), m.end() + 120)
                snippet = page_text[start:end].replace("\n", " ").strip()
                hits.append((page_num, term, snippet))
                break
    return hits


def run():
    print("=" * 78)
    print("CITATION CLAIM VERIFICATION REPORT")
    print("=" * 78)

    for key, claim, terms in CLAIMS:
        pdf_path = os.path.join(CITATIONS_DIR, f"{key}.pdf")
        print(f"\n{'─'*78}")
        print(f"[{key}]")
        print(f"Claim : {claim}")

        if not os.path.exists(pdf_path):
            print("STATUS: MISSING PDF")
            continue

        text_pages, err = extract_text(pdf_path)
        if err:
            print(f"STATUS: PDF READ ERROR — {err}")
            continue
        if not text_pages:
            print("STATUS: NO TEXT EXTRACTED")
            continue

        hits = find_hits(text_pages, terms)
        if hits:
            print(f"STATUS: CONFIRMED ({len(hits)} hit(s))")
            for page_num, matched_term, snippet in hits:
                print(f"  p.{page_num} [{matched_term!r}]")
                print(f"    «...{snippet}...»")
        else:
            # Try full text (up to 20 pages)
            text_pages_full, _ = extract_text(pdf_path, max_pages=20)
            hits2 = find_hits(text_pages_full, terms)
            if hits2:
                print(f"STATUS: CONFIRMED (found in pages 13-20, {len(hits2)} hit(s))")
                for page_num, matched_term, snippet in hits2:
                    print(f"  p.{page_num} [{matched_term!r}]")
                    print(f"    «...{snippet}...»")
            else:
                print("STATUS: NOT FOUND — review manually")
                print(f"  Searched for: {terms}")

    print(f"\n{'='*78}")
    print("END OF REPORT")
    print("=" * 78)


if __name__ == "__main__":
    run()
