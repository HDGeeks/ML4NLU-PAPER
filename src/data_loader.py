"""
data_loader.py

Intent
------
Centralized data loading for multilingual bias experiments.

This module:
- Loads corpus per language
- Loads gender anchors per language
- Loads profession list per language

All language-specific file paths are defined here.

CSV Schemas (team contract)
---------------------------
1) Corpus
   - Plain text file, one sentence per line:
     data/<lang_dir>/corpus_<lang>.txt

2) Anchors
   - CSV with header:
       gender,term
     Example rows:
       male,hombre
       female,mujer

3) Professions
   - CSV with header:
       profession_id,profession_lang
     Example rows:
       doctor,médico
       nurse,enfermera
"""

from __future__ import annotations

import csv
from pathlib import Path


BASE_DIR = Path("data")

_LANG_DIR = {
    "es": "spanish",
    "ar": "arabic",
    "ti": "tigrigna",
    "en": "english"
}


def language_dir(lang: str) -> str:
    try:
        return _LANG_DIR[lang]
    except KeyError as e:
        raise ValueError(f"Unsupported language '{lang}'. Use one of: {sorted(_LANG_DIR)}") from e


def corpus_path(lang: str) -> Path:
    return BASE_DIR / language_dir(lang) / f"corpus_{lang}.txt"


def anchors_path(lang: str) -> Path:
    return BASE_DIR / language_dir(lang) / f"anchors_{lang}.csv"


def professions_path(lang: str) -> Path:
    return BASE_DIR / language_dir(lang) / f"professions_{lang}.csv"


def _read_nonempty_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def load_corpus(lang: str) -> list[str]:
    return _read_nonempty_lines(corpus_path(lang))


def load_anchors(lang: str) -> tuple[list[str], list[str]]:
    """
    Returns (male_terms, female_terms) from anchors CSV.
    Expected header: gender,term
    """
    path = anchors_path(lang)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    male, female = [], []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Empty anchors file: {path}")

        required = {"gender", "term"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Anchors CSV {path} missing columns: {sorted(missing)}")

        for row in reader:
            g = (row.get("gender") or "").strip().lower()
            t = (row.get("term") or "").strip()
            if not t:
                continue
            if g == "male":
                male.append(t)
            elif g == "female":
                female.append(t)

    if not male or not female:
        raise ValueError(
            f"Anchors file {path} must contain at least 1 male and 1 female term "
            f"(found male={len(male)}, female={len(female)})."
        )

    return male, female


def load_professions(lang: str) -> list[str]:
    """
    Returns profession terms from professions CSV.
    Expected header:
      profession_id,profession_lang,profession
    """
    path = professions_path(lang)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    out: list[str] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Empty professions file: {path}")

        required = {"profession_id", "profession_lang", "profession"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Professions CSV {path} missing columns: {sorted(missing)}")

        for row in reader:
            term = (row.get("profession") or "").strip()
            if term:
                out.append(term)

    if not out:
        raise ValueError(f"No profession terms found in: {path}")

    return out