"""
data_loader.py
--------------
Corpus / anchor / profession loading, tokenizer-level sentence search,
and gender-marker sentence classification.
"""

import csv
from pathlib import Path

# Maps the two-letter language code to the data sub-directory name.
_LANG_DIR = {"ti": "tigrigna", "ar": "arabic", "es": "spanish"}

# Tigrigna gender markers for sentence classification (debug reporting).
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


def load_corpus(lang: str) -> list:
    path = Path("data") / _LANG_DIR[lang] / f"corpus_{lang}.txt"
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_anchors(lang: str) -> tuple:
    """Return (male_terms, female_terms) from the anchors CSV."""
    path = Path("data") / _LANG_DIR[lang] / f"anchors_{lang}.csv"
    male, female = [], []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            gender = row["gender"].strip().lower()
            term   = row["term"].strip()
            if gender == "male":
                male.append(term)
            elif gender == "female":
                female.append(term)
    return male, female


def load_professions(lang: str) -> list:
    path = Path("data") / _LANG_DIR[lang] / f"professions_{lang}.csv"
    with open(path, encoding="utf-8", newline="") as f:
        return [row["profession"].strip() for row in csv.DictReader(f) if row["profession"].strip()]


def classify_sentence(sentence: str) -> str:
    """Return 'F', 'M', or 'N' based on gender marker presence."""
    for m in _FEMALE_MARKERS:
        if m in sentence:
            return "F"
    for m in _MALE_MARKERS:
        if m in sentence:
            return "M"
    return "N"


def find_sentences(corpus: list, word: str, n: int, tokenizer, verbose: bool = False) -> list:
    """Return up to n sentences that contain word (tokenizer-level matching)."""
    word_tokens = tokenizer.tokenize(word)
    if verbose:
        print(f"\n  find_sentences('{word}', n={n})")
        print(f"    word tokenizes to: {word_tokens}")

    found = []
    for sentence in corpus:
        sent_tokens = tokenizer.tokenize(sentence)
        for i in range(len(sent_tokens) - len(word_tokens) + 1):
            if sent_tokens[i : i + len(word_tokens)] == word_tokens:
                found.append(sentence)
                break
        if len(found) >= n:
            break

    if verbose:
        print(f"    found {len(found)}/{n} sentences")
        for i, s in enumerate(found):
            print(f"    [{i+1}] {s}")
        if len(found) < n:
            print(f"    WARNING: only {len(found)} sentences — word may be SKIPPED in bulk mode")
    return found


def find_all_matching(corpus: list, word: str) -> list:
    """Return (row_idx, sentence) for every corpus sentence containing word."""
    return [(idx, s) for idx, s in enumerate(corpus) if word in s]

def find_sentences_arabic(corpus: list, word: str, n: int, verbose: bool = False) -> list:
    """Arabic sentence finder using substring matching to handle morphological prefixes."""
    if verbose:
        print(f"\n  find_sentences_arabic('{word}', n={n})")

    found = []
    for sentence in corpus:
        if word in sentence:
            found.append(sentence)
        if len(found) >= n:
            break

    if verbose:
        print(f"    found {len(found)}/{n} sentences")
        for i, s in enumerate(found):
            print(f"    [{i+1}] {s}")
        if len(found) < n:
            print(f"    WARNING: only {len(found)} sentences — word may be SKIPPED in bulk mode")
    return found