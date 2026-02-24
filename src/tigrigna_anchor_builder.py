"""
tigrigna_anchor_builder.py
---------------------------
Creates two files needed for the Tigrigna bias pipeline:

  1. data/tigrigna/anchors_ti.csv
       Columns: gender, type, subtype, term, english_ref
       The 'english_ref' column is for human review only —
       it is never read by data_loader.py or any pipeline code.

  2. data/tigrigna/corpus_ti_anchor_sentences.txt
       4 sample sentences per anchor term.
       Append to corpus_ti.txt before running main.py so every
       anchor term meets the ANCHOR_CONTEXTS = 3 minimum.

ANCHOR TYPES:
  pronoun  / subject    — he / she
  pronoun  / object     — him / her
  pronoun  / possessive — his / her
  kinship  / —          — father, mother, brother, sister …
  generic  / —          — man, woman, boy, girl …

NOTE ON ሰበይቲ:
  This word covers both "wife" (kinship) and "woman" (generic) in Tigrigna.
  It appears in both rows intentionally. This lexical overlap is worth
  noting in the paper (Section 3.2) as a cross-lingual difference.

BEFORE RUNNING:
  Have your native speaker verify:
    1. Every term and its english_ref translation
    2. Sample sentences — especially verb agreement
    3. Flag any term that feels unnatural or dialectally marked

USAGE:
  python src/tigrigna_anchor_builder.py
"""

from __future__ import annotations

from pathlib import Path
import csv


# ══════════════════════════════════════════════════════════════════════════════
#  ANCHOR TERMS
#  Format: (type, subtype, tigrigna_term, english_ref)
#
#  english_ref is DOCUMENTATION ONLY — never used in pipeline code.
#  It exists so you and your native speaker can quickly verify
#  that each Tigrigna term maps to the intended English concept.
# ══════════════════════════════════════════════════════════════════════════════

MALE_ANCHORS = [
    # type        subtype        term           english_ref
    ("pronoun",  "subject",     "ንሱ",          "he"),
    ("pronoun",  "object",      "ንዕኡ",         "him"),
    ("pronoun",  "possessive",  "ናቱ",          "his"),

    ("kinship",  "",            "ኣቦ",          "father"),
    ("kinship",  "",            "ሓው",          "brother"),
    ("kinship",  "",            "ወዲ ሓው",       "nephew"),
    ("kinship",  "",            "በዓል ገዛ",      "husband"),

    ("generic",  "",            "ሰብኣይ",        "man"),
    ("generic",  "",            "ወዲ",          "boy / son"),
    ("generic",  "",            "ተባዕታይ",      "male"),
    ("generic",  "",            "ወዲ",         "young man"),       # ← verify with native speaker
    ("generic",  "",            "ኣቦሓጎ",        "grandfather"),
]

FEMALE_ANCHORS = [
    # type        subtype        term           english_ref
    ("pronoun",  "subject",     "ንሳ",          "she"),
    ("pronoun",  "object",      "ንዓኣ",         "her"),
    ("pronoun",  "possessive",  "ናታ",          "her (possessive)"),

    ("kinship",  "",            "ኣደ",          "mother"),
    ("kinship",  "",            "ሓፍቲ",         "sister"),
    ("kinship",  "",            "ጓል ሓፍቲ",     "niece"),
    ("kinship",  "",            "ሰበይቲ",        "wife"),

    ("generic",  "",            "ሰበይቲ",        "woman"),           # same word as wife — intentional
    ("generic",  "",            "ጓል",          "girl / daughter"),
    ("generic",  "",            "ኣንስተይቲ",     "female"),
    ("generic",  "",            "ጓል",        "young woman"),     # ← NEEDS CORRECTION
    ("generic",  "",            "ዓባየይ",      "grandmother"),
]


# ══════════════════════════════════════════════════════════════════════════════
#  SAMPLE SENTENCES — 4 per anchor term
#  english_ref translations in comments for readability.
#  Native speaker: check verb agreement in every Tigrigna sentence.
# ══════════════════════════════════════════════════════════════════════════════

ANCHOR_SENTENCES: dict[str, list[str]] = {

    # ── Male pronouns ──────────────────────────────────────────────────────────
    "ንሱ": [
        "ንሱ ናብ ቤት ከይዱ።",                    # He went home.
        "ንሱ ጽቡቕ ሰብ እዩ።",                    # He is a good person.
        "ንሱ ስርሑ ወዲኡ።",                       # He finished his work.
        "ንሱ ቅልጡፍ ይሰርሕ።",                    # He works quickly.
    ],
    "ንዕኡ": [
        "ሓጊዝናዮ ንዕኡ።",                        # We helped him.
        "ነጊርናዮ ንዕኡ ኩሉ ነገር።",                # We told him everything.
        "ርኢናዮ ንዕኡ ኣብ ዓዲ።",                  # We saw him in town.
        "ሰዲድናሉ ንዕኡ ደብዳቤ።",                  # We sent him a letter.
    ],
    "ናቱ": [
        "ስርሑ ናቱ እዩ።",                        # His work is his.
        "ቤቱ ናቱ ዓቢ እዩ።",                     # His house is big.
        "ናቱ ሓሳብ ኣገዳሲ እዩ።",                  # His idea is important.
        "ናቱ ውሳነ ትኽክል እዩ።",                  # His decision is correct.
    ],

    # ── Female pronouns ────────────────────────────────────────────────────────
    "ንሳ": [
        "ንሳ ናብ ቤት ከይዳ።",                    # She went home.
        "ንሳ ጽቡቕ ሰብ እያ።",                    # She is a good person.
        "ንሳ ስርሓ ወዲኣ።",                       # She finished her work.
        "ንሳ ቅልጥፍቲ ትሰርሕ።",                   # She works quickly.
    ],
    "ንዓኣ": [
        "ሓጊዝናያ ንዓኣ።",                        # We helped her.
        "ነጊርናያ ንዓኣ ኩሉ ነገር።",                # We told her everything.
        "ርኢናያ ንዓኣ ኣብ ዓዲ።",                  # We saw her in town.
        "ሰዲድናላ ንዓኣ ደብዳቤ።",                  # We sent her a letter.
    ],
    "ናታ": [
        "ስርሓ ናታ እዩ።",                        # Her work is hers.
        "ቤታ ናታ ዓቢ እዩ።",                     # Her house is big.
        "ናታ ሓሳብ ኣገዳሲ እዩ።",                  # Her idea is important.
        "ናታ ውሳነ ትኽክል እዩ።",                  # Her decision is correct.
    ],

    # ── Male kinship ───────────────────────────────────────────────────────────
    "ኣቦ": [
        "ኣቦ ናብ ስራሕ ከይዱ።",                   # Father went to work.
        "ኣቦ ጽቡቕ ሰብ እዩ።",                    # Father is a good person.
        "ኣቦ ቆልዑ ይሕግዝ።",                     # Father helps the children.
        "ኣቦ ኣብ ከተማ ይሰርሕ።",                  # Father works in the city.
    ],
    "ሓው": [
        "ሓወይ ትምህርቲ ዛዚሙ።",                   # My brother finished school.
        "ሓው ናብ ዓዲ ተመሊሱ።",                   # Brother returned to town.
        "ሓወይ ጎበዝ ሰብ እዩ።",                   # My brother is a capable person.
        "ሓው ምስ ኣቦ ይሰርሕ።",                   # Brother works with father.
    ],
    "ወዲ ሓው": [
        "ወዲ ሓወይ ናብ ቤት ትምህርቲ ከይዱ።",         # My nephew went to school.
        "ወዲ ሓወይ ቅኑዕ ሰብ እዩ።",               # My nephew is an upright person.
        "ወዲ ሓው ስርሑ ጀሚሩ።",                   # The nephew started his work.
        "ወዲ ሓወይ ኣብ ከተማ ይሰርሕ።",             # My nephew works in the city.
    ],
    "በዓል ገዛ": [
        "በዓል ገዛ ናብ ዕዳጋ ከይዱ።",               # The husband went to market.
        "በዓል ገዛ ምሸት ተመሊሱ።",                 # The husband returned in the evening.
        "በዓል ገዛ ንስድራ ይሕግዝ።",                # The husband helps the family.
        "በዓል ገዛ ጽቡቕ ሰብ እዩ።",                # The husband is a good person.
    ],

    # ── Female kinship ─────────────────────────────────────────────────────────
    "ኣደ": [
        "ኣደ ኣብ ቤት ትሰርሕ።",                   # Mother works at home.
        "ኣደ ጽቡቕ ሰብ እያ።",                    # Mother is a good person.
        "ኣደ ቆልዑ ትሕግዝ።",                     # Mother helps the children.
        "ኣደ ምግቢ ኣዳልያ።",                     # Mother prepared food.
    ],
    "ሓፍቲ": [
        "ሓፍተይ ትምህርቲ ዛዚማ።",                  # My sister finished school.
        "ሓፍቲ ናብ ዓዲ ተመሊሳ።",                  # Sister returned to town.
        "ሓፍተይ ጎቦዝቲ ሰብ እያ።",                # My sister is a capable person.
        "ሓፍቲ ምስ ኣደ ትሰርሕ።",                  # Sister works with mother.
    ],
    "ጓል ሓፍቲ": [
        "ጓል ሓፍተይ ናብ ቤት ትምህርቲ ከይዳ።",        # My niece went to school.
        "ጓል ሓፍተይ ቅንእቲ ሰብ እያ።",             # My niece is a virtuous person.
        "ጓል ሓፍቲ ስርሓ ጀሚራ።",                  # The niece started her work.
        "ጓል ሓፍተይ ኣብ ከተማ ትሰርሕ።",            # My niece works in the city.
    ],
    "ሰበይቲ": [
        "ሰበይቲ ናብ ዕዳጋ ከይዳ።",                 # The woman/wife went to market.
        "ሰበይቲ ምሸት ተመሊሳ።",                   # The woman/wife returned in the evening.
        "ሰበይቲ ንስድራ ትሕግዝ።",                  # The woman/wife helps the family.
        "ሰበይቲ ጽቡቕ ሰብ እያ።",                  # The woman/wife is a good person.
    ],

    # ── Male generic ───────────────────────────────────────────────────────────
    "ሰብኣይ": [
        "ሰብኣይ ኣብ ቤት ሰሪሑ።",                  # The man worked at home.
        "ሰብኣይ ናብ ዕዳጋ ከይዱ።",                 # The man went to market.
        "ሰብኣይ ንስድራ ይሕግዝ።",                  # The man helps the family.
        "ሰብኣይ ጽቡቕ ሰብ እዩ።",                  # The man is a good person.
    ],
    "ወዲ": [
        "ወዲ ናብ ቤት ትምህርቲ ከይዱ።",             # The boy went to school.
        "ወዲ ኣብ ጎደና ይጻወት።",                  # The boy plays in the street.
        "ወዲ ንኣደኡ ሓጊዙ።",                     # The boy helped his mother.
        "ወዲ ዓቢ ሰብ ክኸውን ይደሊ።",              # The boy wants to become great.
    ],
    "ተባዕታይ": [
        "ተባዕታይ ሰብ ኣብ ስራሕ ጎበዝ እዩ።",         # A male person is skilled at work.
        "ተባዕታይ ብሓይሉ ይሰርሕ።",                # The male works with his strength.
        "ተባዕታይ ሓላፍነቱ ይፈልጥ።",               # The male knows his responsibility.
        "ተባዕታይ ናብ ዕዳጋ ወሲዱ።",               # The male went to the market.
    ],
    "ወዳም": [
        "ወዳም ኣብ ዓዲ ፍሉጥ እዩ።",               # The young man is known in town.
        "ወዳም ብጹዕ ናብ ስራሕ ኸይዱ።",             # The young man went eagerly to work.
        "ወዳም ሓዳር ኣቑሙ።",                     # The young man established a home.
        "ወዳም ብዙሕ ዓቕሚ ኣለዎ።",                # The young man has much ability.
    ],
    "ኣቦሓጎ": [
        "ኣቦሓጎ ተሞክሮ ኣለዎ።",                   # The grandfather has experience.
        "ኣቦሓጎ ንቆልዑ ዕድሜኦም ይነግሮም።",          # The grandfather tells the children stories.
        "ኣቦሓጎ ኣብ ዓዲ ይነብር።",                 # The grandfather lives in the village.
        "ኣቦሓጎ ጥበብ ዘለዎ ሰብ እዩ።",             # The grandfather is a wise person.
    ],

    # ── Female generic ─────────────────────────────────────────────────────────
    "ጓል": [
        "ጓል ናብ ቤት ትምህርቲ ከይዳ።",             # The girl went to school.
        "ጓል ኣብ ጎደና ትጻወት።",                  # The girl plays in the street.
        "ጓል ንኣደኣ ሓጊዛ።",                     # The girl helped her mother.
        "ጓል ዓቢ ሰብ ክትከውን ትደሊ።",             # The girl wants to become great.
    ],
    "ኣንስተይቲ": [
        "ኣንስተይቲ ሰብ ኣብ ስራሕ ጎቦዝቲ እያ።",       # A female person is skilled at work.
        "ኣንስተይቲ ብትዕግስቲ ትሰርሕ።",             # The female works with patience.
        "ኣንስተይቲ ሓላፍነታ ትፈልጥ።",              # The female knows her responsibility.
        "ኣንስተይቲ ናብ ዕዳጋ ወሲዳ።",              # The female went to the market.
    ],
    "ኣደወልዲ": [
        "ኣደወልዲ ተሞክሮ ኣሎዋ።",                  # The grandmother has experience.
        "ኣደወልዲ ንቆልዑ ዛንታ ትነግሮም።",           # The grandmother tells the children stories.
        "ኣደወልዲ ኣብ ዓዲ ትነብር።",                # The grandmother lives in the village.
        "ኣደወልዲ ጥበብ ዘሎዋ ሰብ እያ።",            # The grandmother is a wise person.
    ],

    # ── Placeholder — awaiting native speaker correction ──────────────────────
    "??? ጓሉም WRONG — replace with native speaker input": [
        "??? sentence needed after correction",
        "??? sentence needed after correction",
        "??? sentence needed after correction",
        "??? sentence needed after correction",
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
#  Output paths
# ══════════════════════════════════════════════════════════════════════════════

ANCHORS_CSV          = Path("data/tigrigna/anchors_ti.csv")
ANCHOR_SENTENCES_TXT = Path("data/tigrigna/corpus_ti_anchor_sentences.txt")


# ── Validation ────────────────────────────────────────────────────────────────

def validate() -> bool:
    """
    Run basic checks before writing anything.
    Catches common mistakes early so you don't run the full pipeline
    on a broken anchor set.
    """
    ok = True

    if len(MALE_ANCHORS) != len(FEMALE_ANCHORS):
        print(f"  ✗ Unequal anchor counts: "
              f"male={len(MALE_ANCHORS)}, female={len(FEMALE_ANCHORS)}")
        print("    The gender centroids will be skewed. Fix before continuing.")
        ok = False

    # Check every term has sample sentences
    all_terms = [t for _, _, t, _ in MALE_ANCHORS + FEMALE_ANCHORS]
    for term in all_terms:
        if term not in ANCHOR_SENTENCES:
            print(f"  ✗ No sample sentences for: '{term}'")
            ok = False
        elif any("???" in s for s in ANCHOR_SENTENCES[term]):
            print(f"  ⚠ Placeholder sentences still present for: '{term}'")
            print("    Replace with real Tigrigna sentences before running main.py")
            ok = False

    # Check for placeholder terms
    for _, _, term, english_ref in MALE_ANCHORS + FEMALE_ANCHORS:
        if "???" in term:
            print(f"  ⚠ Placeholder term still present: '{term}' ({english_ref})")
            print("    Ask your native speaker for the correct Tigrigna term.")
            ok = False

    return ok


# ── Writers ───────────────────────────────────────────────────────────────────

def write_anchors_csv() -> None:
    """
    Write anchors_ti.csv.

    Columns written:
      gender       — male / female          (used by pipeline)
      type         — pronoun / kinship / generic  (used by pipeline)
      subtype      — subject / object / possessive / ""  (used by pipeline)
      term         — Tigrigna term          (used by pipeline)
      english_ref  — English translation    (DOCUMENTATION ONLY — never read by code)
    """
    ANCHORS_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for anchor_type, subtype, term, english_ref in MALE_ANCHORS:
        rows.append({"gender": "male",   "type": anchor_type,
                     "subtype": subtype, "term": term,
                     "english_ref": english_ref})
    for anchor_type, subtype, term, english_ref in FEMALE_ANCHORS:
        rows.append({"gender": "female", "type": anchor_type,
                     "subtype": subtype, "term": term,
                     "english_ref": english_ref})

    with ANCHORS_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["gender", "type", "subtype", "term", "english_ref"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"✓ Wrote {len(rows)} anchor terms → {ANCHORS_CSV}")
    print(f"  Male:   {len(MALE_ANCHORS)}")
    print(f"  Female: {len(FEMALE_ANCHORS)}")


def write_anchor_sentences() -> None:
    """Write 4 sample sentences per anchor into a standalone file."""
    all_sentences = []
    for _, _, term, _ in MALE_ANCHORS + FEMALE_ANCHORS:
        if term in ANCHOR_SENTENCES:
            all_sentences.extend(ANCHOR_SENTENCES[term])

    ANCHOR_SENTENCES_TXT.parent.mkdir(parents=True, exist_ok=True)
    ANCHOR_SENTENCES_TXT.write_text("\n".join(all_sentences), encoding="utf-8")
    print(f"✓ Wrote {len(all_sentences)} anchor sentences → {ANCHOR_SENTENCES_TXT}")


def print_review_table() -> None:
    """Print a clean table for native speaker review."""
    print("\n" + "=" * 80)
    print("ANCHOR REVIEW TABLE — share with native speaker")
    print("Note: english_ref column is for review only, never used in code")
    print("=" * 80)
    print(f"  {'GENDER':<8} {'TYPE':<10} {'SUBTYPE':<12} {'TERM':<25} {'ENGLISH REF':<18}")
    print(f"  {'-'*8} {'-'*10} {'-'*12} {'-'*25} {'-'*18}")

    for gender, anchors in [("male", MALE_ANCHORS), ("female", FEMALE_ANCHORS)]:
        for anchor_type, subtype, term, english_ref in anchors:
            flag = " ⚠" if "???" in term else ""
            print(f"  {gender:<8} {anchor_type:<10} {(subtype or '—'):<12}"
                  f" {term:<25} {english_ref:<18}{flag}")
        print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("Tigrigna Anchor Builder")
    print("=" * 80)

    valid = validate()
    if not valid:
        print("\n  Fix the issues above before running main.py.")
        print("  The files will still be written so you can inspect them.\n")

    write_anchors_csv()
    write_anchor_sentences()
    print_review_table()

    print("NEXT STEPS:")
    print("  1. Review the table above with your native speaker")
    print("  2. Replace any ⚠ placeholder terms and re-run this script")
    print("  3. Append anchor sentences to the main corpus:")
    print("       cat data/tigrigna/corpus_ti_anchor_sentences.txt"
          " >> data/tigrigna/corpus_ti.txt")
    print("  4. Run main.py")


if __name__ == "__main__":
    main()