"""
tigrigna_corpus_builder.py
---------------------------
Generates a synthetic Tigrigna corpus for the gender bias pipeline.

WHY SYNTHETIC:
  Tigrigna is a low-resource language. Template generation is the standard
  academic approach and is fully defensible when disclosed in Section 3.2.

KEY LINGUISTIC NOTE — TIGRIGNA VERB AGREEMENT:
  Profession terms are grammatically neutral (same word for male and female).
  Gender is expressed through verb conjugation. Templates are therefore
  defined as (male_version, female_version) PAIRS — the whole sentence,
  pronoun AND verb, is written together so they always agree.

CORPUS DESIGN — ANCHOR-PROFESSION INTEGRATION:
  The corpus has two layers:

  Layer 1 — PROFESSION TEMPLATES (existing):
    General sentences about profession words in varied grammatical positions.
    These build rich contextual representations of each profession term.

  Layer 2 — ANCHOR-PROFESSION TEMPLATES (new):
    Every anchor term appears in the same sentence as every profession term.
    e.g. "ኣቦ ሓኪም እዩ ንሱ ኣብ ሆስፒታል ይሰርሕ།"
         "ሓፍተይ ነርስ እያ ንሳ ንሕሙማት ትሕግዝ།"

    This means:
      - The model's representation of each anchor is shaped by professional context
      - Every anchor term will appear alongside every profession term
      - The gender direction is built from anchor vectors that have seen
        professional language — making it directly relevant to occupational bias
      - This is methodologically novel for contextual multilingual bias work

  SENTENCE COUNT:
    Layer 1: ~2000 sentences  (profession templates × 30 professions)
    Layer 2: 24 anchors × 30 professions × 2 genders = 1,440 sentences
    Total:   ~3,440 sentences

BEFORE RUNNING:
  Have your native speaker review:
    1. Every ANCHOR-PROFESSION template pair (most important — new addition)
    2. Every PROFESSION template pair for verb agreement
    3. The PROFESSIONS_TI list — must match professions_ti.csv exactly

USAGE:
  python src/tigrigna_corpus_builder.py
"""

from __future__ import annotations

import random
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
#  ANCHOR-PROFESSION TEMPLATES
#  Each template places an anchor term and a profession term in the same
#  sentence. {anchor} and {prof} are both filled at generation time.
#
#  Format: tuple (male_version, female_version)
#  Both versions must use the SAME anchor term — only verb/pronoun changes.
#
#  These are grouped by anchor TYPE so your native speaker can review
#  all pronoun templates together, all kinship together, etc.
# ══════════════════════════════════════════════════════════════════════════════

# ── Pronoun (subject) anchor templates ───────────────────────────────────────
# Anchor = ንሱ (he) / ንሳ (she)
# The pronoun refers back to the profession word in the prior clause.

ANCHOR_PROF_PRONOUN_SUBJ = [
    (
        "{anchor_m} {prof} እዩ ንሱ ኣብ ስርሑ ጎበዝ እዩ",
        "{anchor_f} {prof} እያ ንሳ ኣብ ስርሓ ጎቦዝቲ እያ",
    ),
    # M: "He is a {prof}. He is skilled at his work."
    # F: "She is a {prof}. She is skilled at her work."

    (
        "{anchor_m} {prof} እዩ ንሱ ንህዝቢ ኣገልግሎት ይህብ",
        "{anchor_f} {prof} እያ ንሳ ንህዝቢ ኣገልግሎት ትህብ",
    ),
    # M: "He is a {prof}. He serves the public."
    # F: "She is a {prof}. She serves the public."

    (
        "{anchor_m} {prof} እዩ ንሱ ኩሉ መዓልቲ ይጽዕር",
        "{anchor_f} {prof} እያ ንሳ ኩሉ መዓልቲ ትጽዕር",
    ),
    # M: "He is a {prof}. He works hard every day."
    # F: "She is a {prof}. She works hard every day."
]

# ── Pronoun (object) anchor templates ────────────────────────────────────────
# Anchor = ንዕኡ (him) / ንዓኣ (her)

ANCHOR_PROF_PRONOUN_OBJ = [
    (
        "ሓደ {prof} ረኸብና ሓጊዝናዮ ንዕኡ",
        "ሓንቲ {prof} ረኸብና ሓጊዝናያ ንዓኣ",
    ),
    # M: "We found a {prof}. We helped him."
    # F: "We found a {prof}. We helped her."

    (
        "{prof} ናብ ቤት ጸዊዕናዮ ንዕኡ ብዙሕ ሕቶ ሓተትናዮ",
        "{prof} ናብ ቤት ጸዊዕናያ ንዓኣ ብዙሕ ሕቶ ሓተትናያ",
    ),
    # M: "We invited the {prof} home, him. We asked him many questions."
    # F: "We invited the {prof} home, her. We asked her many questions."
]

# ── Pronoun (possessive) anchor templates ────────────────────────────────────
# Anchor = ናቱ (his) / ናታ (her)

ANCHOR_PROF_PRONOUN_POSS = [
    (
        "{prof} ናቱ ስራሕ ብትዕግስቲ ይሰርሕ",
        "{prof} ናታ ስራሕ ብትዕግስቲ ትሰርሕ",
    ),
    # M: "The {prof} works his work with patience."
    # F: "The {prof} works her work with patience."

    (
        "{prof} ናቱ ሓላፍነት ይፈልጥ",
        "{prof} ናታ ሓላፍነት ትፈልጥ",
    ),
    # M: "The {prof} knows his responsibility."
    # F: "The {prof} knows her responsibility."
]

# ── Kinship anchor templates ──────────────────────────────────────────────────
# Anchor = ኣቦ/ኣደ, ሓው/ሓፍቲ, ወዲ ሓው/ጓል ሓፍቲ, በዓል ገዛ/ሰበይቲ
# The kinship term identifies a person who IS the profession.

ANCHOR_PROF_KINSHIP = [
    (
        "ኣቦይ {prof} እዩ ንሱ ስርሑ ይፈቱ",
        "ኣደይ {prof} እያ ንሳ ስርሓ ትፈቱ",
    ),
    # M: "My father is a {prof}. He loves his work."
    # F: "My mother is a {prof}. She loves her work."

    (
        "ሓወይ {prof} እዩ ንሱ ብዙሕ ተሞክሮ ኣለዎ",
        "ሓፍተይ {prof} እያ ንሳ ብዙሕ ተሞክሮ ኣሎዋ",
    ),
    # M: "My brother is a {prof}. He has much experience."
    # F: "My sister is a {prof}. She has much experience."

    (
        "ወዲ ሓወይ {prof} እዩ ስርሑ ኣብ ከተማ እዩ",
        "ጓል ሓፍተይ {prof} እያ ስርሓ ኣብ ከተማ እዩ",
    ),
    # M: "My nephew is a {prof}. His work is in the city."
    # F: "My niece is a {prof}. Her work is in the city."

    (
        "በዓል ገዛይ {prof} እዩ ንሱ ጽቡቕ ሰብ እዩ",
        "ሰበይተይ {prof} እያ ንሳ ጽቡቕ ሰብ እያ",
    ),
    # M: "My husband is a {prof}. He is a good person."
    # F: "My wife is a {prof}. She is a good person."

    (
        "ኣቦይ {prof} ነበረ ነቲ ሞያ ብዝተፈለየ ኣገባብ ይሰርሖ ነበረ",
        "ኣደይ {prof} ነበረት། ነቲ ሞያ ብዝተፈለየ ኣገባብ ትሰርሖ ነበረት",
    ),
    # M: "My father was a {prof}. He practiced the profession uniquely."
    # F: "My mother was a {prof}. She practiced the profession uniquely."

    (
        "ሓወይ {prof} ክኸውን ይደሊ ብዙሕ ጻዕሪ ይጽዕር",
        "ሓፍተይ {prof} ክትከውን ትደሊ ብዙሕ ጻዕሪ ትጽዕር",
    ),
    # M: "My brother wants to become a {prof}. He works very hard."
    # F: "My sister wants to become a {prof}. She works very hard."
]

# ── Generic anchor templates ──────────────────────────────────────────────────
# Anchor = ሰብኣይ/ሰበይቲ, ወዲ/ጓል, ተባዕታይ/ኣንስተይቲ, ወዲ(young)/ጓል(young), ኣቦሓጎ/ዓባየይ

ANCHOR_PROF_GENERIC = [
    (
        "ሰብኣይ {prof} ኮይኑ ይሰርሕ ንሱ ኩሉ ሰብ ዘፍቅሮ እዩ",
        "ሰበይቲ {prof} ኮይና ትሰርሕ ንሳ ኩሉ ሰብ ዘፍቅሮ እያ",
    ),
    # M: "A man works as a {prof}. He is loved by everyone."
    # F: "A woman works as a {prof}. She is loved by everyone."

    (
        "ወዲ {prof} ኮይኑ ክሰርሕ ይደሊ ብዙሕ ተስፋ ኣለዎ",
        "ጓል {prof} ኮይና ክትሰርሕ ትደሊ ብዙሕ ተስፋ ኣሎዋ",
    ),
    # M: "A boy wants to work as a {prof}. He has much hope."
    # F: "A girl wants to work as a {prof}. She has much hope."

    (
        "ተባዕታይ {prof} ምዃን ዓቢ ሕልሚ ኣለዎ ንሱ ብትግሃት ይጽዕር",
        "ኣንስተይቲ {prof} ምዃን ዓቢ ሕልሚ ኣሎዋ ንሳ ብትግሃት ትጽዕር",
    ),
    # M: "A male has a big dream of becoming a {prof}. He strives diligently."
    # F: "A female has a big dream of becoming a {prof}. She strives diligently."

    (
        "ወዲ {prof} ኮይኑ ዓቢ ተስፋ ኣለዎ ትምህርቱ ዛዚሙ ይሰርሕ",
        "ጓል {prof} ኮይና ዓቢ ተስፋ ኣሎዋ ትምህርታ ዛዚማ ትሰርሕ",
    ),
    # M: "A young man has great hope of being a {prof}. Finished studies, now works."
    # F: "A young woman has great hope of being a {prof}. Finished studies, now works."

    (
        "ኣቦሓጎ {prof} ነበረ ንሱ ኣብ ዓዲ ፍሉጥ ነበረ",
        "ዓባየይ {prof} ነበረት ንሳ ኣብ ዓዲ ፍልጥቲ ነበረት",
    ),
    # M: "The grandfather was a {prof}. He was well known in the village."
    # F: "The grandmother was a {prof}. She was well known in the village."
]

# All anchor-profession templates combined
ALL_ANCHOR_PROF_TEMPLATES = (
    ANCHOR_PROF_PRONOUN_SUBJ
    + ANCHOR_PROF_PRONOUN_OBJ
    + ANCHOR_PROF_PRONOUN_POSS
    + ANCHOR_PROF_KINSHIP
    + ANCHOR_PROF_GENERIC
)

# Paired anchor terms — (male_form, female_form) — must match anchors_ti.csv
# Each pair maps to the {anchor_m} / {anchor_f} placeholders above
ANCHOR_PAIRS = [
    # pronouns
    ("ንሱ",       "ንሳ"),        # he / she   (subject)
    ("ንዕኡ",      "ንዓኣ"),       # him / her  (object)
    ("ናቱ",       "ናታ"),        # his / her  (possessive)
    # kinship
    ("ኣቦ",       "ኣደ"),        # father / mother
    ("ሓው",       "ሓፍቲ"),       # brother / sister
    ("ወዲ ሓው",   "ጓል ሓፍቲ"),   # nephew / niece
    ("በዓል ገዛ",  "ሰበይቲ"),      # husband / wife
    # generic
    ("ሰብኣይ",    "ሰበይቲ"),      # man / woman
    ("ወዲ",       "ጓል"),        # boy / girl
    ("ተባዕታይ",  "ኣንስተይቲ"),   # male / female
    ("ወዲ",       "ጓል"),        # young man / young woman
    ("ኣቦሓጎ",    "ዓባየይ"),      # grandfather / grandmother
]


# ══════════════════════════════════════════════════════════════════════════════
#  PROFESSION TEMPLATES  (Layer 1 — unchanged from before)
# ══════════════════════════════════════════════════════════════════════════════

TEMPLATES_SUBJECT = [
    "{prof} ኣብ ከተማ ይሰርሕ",
    "{prof} ንህዝቢ ኣገልግሎት ይህብ",
    "{prof} ኣብ ስርሑ ብጣዕሚ ጎበዝ እዩ",
    "{prof} ኩሉ መዓልቲ ይጽዕር",
    "{prof} ናይ ስርሑ ሓላፍነት ይስከም",
    "{prof} ኣብ ሓድሽ ፕሮጀክት ይሳተፍ",
    "{prof} ምስ ብዙሕ ሰባት ይሰርሕ",
    (
        "{prof} ናብ ቤት ጽሕፈት ኣተወ ንሱ ስርሑ ጀሚሩ",
        "{prof} ናብ ቤት ጽሕፈት ኣተወት ንሳ ስርሓ ጀሚራ",
    ),
    (
        "{prof} ኣብ ኣኼባ ቀሪቡ ንሱ ርኢቱ ብዝርዝር ገለጸ",
        "{prof} ኣብ ኣኼባ ቀሪባ ንሳ ርኢታ ብዝርዝር ገለጸት",
    ),
    (
        "{prof} ሓደ ሓድሽ ፕሮጀክት ጀሚሩ ንሱ ምስ ጋንታኡ ይሰርሕ",
        "{prof} ሓደ ሓድሽ ፕሮጀክት ጀሚራ ንሳ ምስ ጋንታኣ ትሰርሕ",
    ),
    (
        "{prof} ናይ ዓሚል ሕቶ ሰምዐ ንሱ ድሕሪ ምምርማር መልሲ ሃበ",
        "{prof} ናይ ዓሚል ሕቶ ሰምዐት ንሳ ድሕሪ ምምርማር መልሲ ሃበት",
    ),
    (
        "{prof} ምሸት ሰሪሑ ንሱ ዝተሓሰበሉ ዕዮ ወዲኡ",
        "{prof} ምሸት ሰሪሓ ንሳ ዝተሓሰበቶ ዕዮ ወዲኣ",
    ),
    (
        "{prof} ዕዮ ኣዳልዩ ቀረበ ንሱ ኩሉ ሰብ ዘደንቕ ስራሕ ሰሪሑ",
        "{prof} ዕዮ ኣዳልያ ቀረበት ንሳ ኩሉ ሰብ ዘደንቕ ስራሕ ሰሪሓ",
    ),
    (
        "{prof} ካብ ርሑቕ ናብ ስርሑ መጸ ንሱ ጽቡቕ ስምዒት ኣሎዎ",
        "{prof} ካብ ርሑቕ ናብ ስርሓ መጸት ንሳ ጽቡቕ ስምዒት ኣሎዋ",
    ),
]

TEMPLATES_OBJECT = [
    "ሰባት ናብ {prof} ይኸዱ",
    "ሓገዝ ካብ {prof} ረኸብና",
    "ናይ {prof} ምኽሪ ሓተትና",
    "ካብ {prof} ብዙሕ ተምሂርና",
    "ናይ {prof} ርኢቶ ኣገዳሲ እዩ",
    "ምስ {prof} ሓቢርና ሰሪሕና",
    "ናብ {prof} ምኻድ ኣድላዪ እዩ",
    (
        "ሓደ {prof} ረኸብና ኩሉ ጸገምና ነጊርናዮ",
        "ሓንቲ {prof} ረኸብና ኩሉ ጸገምና ነጊርናያ",
    ),
    (
        "{prof} ናብ ቤት ጸዊዕናዮ ኩሉ ሰብ ምስኡ ይዘራረብ ነበረ",
        "{prof} ናብ ቤት ጸዊዕናያ ኩሉ ሰብ ምስኣ ይዘራረብ ነበረ",
    ),
    (
        "ምስ {prof} ተራኸብና ሕቶታትና ብትዕግስቲ ሰሚዑና",
        "ምስ {prof} ተራኸብና ሕቶታትና ብትዕግስቲ ሰሚዓትና",
    ),
    (
        "{prof} ምስ ረኸብናዮ ተሓጎስና ኣብ ስርሑ ክኢላ እዩ",
        "{prof} ምስ ረኸብናያ ተሓጎስና ኣብ ስርሓ ክኢላ እያ",
    ),
    (
        "ናብ {prof} ሓተትና ብሕጊ መልሲ ሃበና",
        "ናብ {prof} ሓተትና། ብሕጊ መልሲ ሃበትና",
    ),
]

TEMPLATES_CONTEXT = [
    "ኣብ ዓዲና ብዙሕ {prof} ኣሎ",
    "ሃገር ብዙሕ {prof} ትደሊ",
    "ሞያ {prof} ኣብ ሕብረተሰብ ኣገዳሲ እዩ",
    "{prof} ምዃን ብዙሕ ጻዕሪ ይሓትት",
    "ደቂ ኣንስትዮ {prof} ኮይነን ይሰርሓ",
    "ደቂ ተባዕትዮ {prof} ኮይኖም ይሰርሑ",
    "ቆልዑ {prof} ምዃን ይምኞቱ",
    "ዕቤት ሃገር ናብ {prof} ይምርኮስ",
    "ሕጂ ብዙሕ ሰብ ናይ {prof} ሞያ ይመርጽ",
    "ናይ {prof} ፍልጠት ናብ ዕቤት ይመርሕ",
    (
        "ሓደ ፍሉጥ {prof} ኣሎ ኣብ ዓዲ ብዙሕ ዝፍለጥ እዩ",
        "ሓንቲ ፍልጥቲ {prof} ኣላ ኣብ ዓዲ ብዙሕ እትፍለጥ እያ",
    ),
    (
        "ናይ ቀርባ ዓርከይ {prof} እዩ ኣዝዩ ሓያሽ ሰብ እዩ",
        "ናይ ቀርባ ዓርኪተይ {prof} እያ ኣዝያ ሓያሽ ሰብ እያ",
    ),
    (
        "ጎረቤትና {prof} እዩ། ንሰባት ኩሉ ግዜ ይሕግዝ།",
        "ጎረቤትና {prof} እያ ንሰባት ኩሉ ግዜ ትሕግዝ",
    ),
    (
        "ሓደ ሓድሽ {prof} ናብ ዓዲ መጸ ብዙሕ ተስፋ ዘሎዎ ሰብ እዩ",
        "ሓንቲ ሓዳስ {prof} ናብ ዓዲ መጸት ብዙሕ ተስፋ ዘሎዋ ሰብ እያ",
    ),
    (
        "ኣቦይ {prof} ነበረ ነቲ ሞያ ብዝተፈለየ ኣገባብ ይሰርሖ ነበረ",
        "ኣደይ {prof} ነበረት། ነቲ ሞያ ብዝተፈለየ ኣገባብ ትሰርሖ ነበረት",
    ),
    (
        "ሓፍተይ {prof} ትኸውን ትደሊ ብዛዕባ እዚ ሞያ ብዙሕ ትሓስብ",
        "ሓወይ {prof} ይኸውን ይደሊ ብዛዕባ እዚ ሞያ ብዙሕ ይሓስብ",
    ),
    (
        "ኣብ ዓዲና ዘሎ {prof} ምስ ኩሉ ሰብ ጽቡቕ ዝምድና ኣሎዎ ፍቱው ሰብ እዩ",
        "ኣብ ዓዲና ዘላ {prof} ምስ ኩሉ ሰብ ጽቡቕ ዝምድና ኣሎዋ ፍትውቲ ሰብ እያ",
    ),
]

ALL_PROFESSION_TEMPLATES = TEMPLATES_SUBJECT + TEMPLATES_OBJECT + TEMPLATES_CONTEXT


# ══════════════════════════════════════════════════════════════════════════════
#  PROFESSION TERMS
# ══════════════════════════════════════════════════════════════════════════════

PROFESSIONS_TI = [
    "ሓኪም",        # doctor
    "ነርስ",         # nurse
    "መምህር",       # teacher
    "ፕሮፌሰር",      # professor
    "መሃንዲስ",      # engineer
    "ሳይንቲስት",     # scientist
    "ጠበቓ",        # lawyer
    "ዳኛ",          # judge
    "ሓላፊ",        # manager
    "ዳይረክተር",    # director
    "ፖሊስ",        # police officer
    "ወተሃደር",      # soldier
    "ሓረስታይ",     # farmer
    "ነጋዳይ",       # trader
    "መካኒክ",       # mechanic
    "ኤለክትሪሻን",   # electrician
    "ፈለኛሞ",       # carpenter
    "ነዳቓይ",       # builder
    "ኣርኪቴክት",    # architect
    "ኣካውንታንት",   # accountant
    "ባንከር",       # banker
    "ስነ ጥበባዊ",   # artist
    "ሙዚቀኛ",      # musician
    "ተዋሳኢ",      # actor
    "ስራሕ ፈጣሪ",  # entrepreneur
    "ኮኮ",          # chef
    "ሰራሕተኛ ጽሬት", # cleaner
    "ሓላዊ ጸጥታ",   # security guard
    "ተክኒሻን",     # technician
    "ፕሮግራመር",   # programmer
]


# ══════════════════════════════════════════════════════════════════════════════
#  Settings
# ══════════════════════════════════════════════════════════════════════════════

TARGET_PROFESSION_SENTENCES = 2000   # Layer 1
OUTPUT_PATH = Path("data/tigrigna/corpus_ti.txt")
RANDOM_SEED = 42


# ── Helpers ───────────────────────────────────────────────────────────────────

def expand_prof_template(template, prof: str, rng: random.Random) -> str:
    """Fill {prof} in a profession template (str or gendered tuple)."""
    if isinstance(template, tuple):
        chosen = rng.choice(template)
    else:
        chosen = template
    return chosen.replace("{prof}", prof)


def generate_anchor_profession_sentences(
    anchor_prof_templates: list[tuple],
    anchor_pairs: list[tuple],
    professions: list[str],
    rng: random.Random,
) -> list[str]:
    """
    Layer 2: Generate one sentence per (anchor_pair, profession, template).

    For each anchor pair (male_form, female_form) × each profession × each
    template, we generate BOTH the male and female version. This ensures:
      - Every anchor term co-occurs with every profession term
      - Male and female versions are always balanced (same count)
      - The gender direction is built from anchors that have seen
        professional context
    """
    sentences = []
    for anchor_m, anchor_f in anchor_pairs:
        for prof in professions:
            for male_tmpl, female_tmpl in anchor_prof_templates:
                # Fill anchor placeholders — templates use {anchor_m}/{anchor_f}
                # only where the anchor is a pronoun/generic standalone term.
                # Kinship templates have the anchor baked in — we still fill
                # {prof} in both versions.
                male_sent = male_tmpl.replace("{anchor_m}", anchor_m)\
                                     .replace("{anchor_f}", anchor_m)\
                                     .replace("{prof}", prof)
                female_sent = female_tmpl.replace("{anchor_f}", anchor_f)\
                                         .replace("{anchor_m}", anchor_f)\
                                         .replace("{prof}", prof)
                sentences.append(male_sent)
                sentences.append(female_sent)

    rng.shuffle(sentences)
    return sentences


def generate_profession_sentences(
    professions: list[str],
    templates: list,
    target_total: int,
    rng: random.Random,
) -> list[str]:
    """Layer 1: Generate profession template sentences."""
    sentences_per_prof = target_total // len(professions)
    remainder = target_total % len(professions)
    all_sentences: list[str] = []

    for i, prof in enumerate(professions):
        candidates = [expand_prof_template(t, prof, rng) for t in templates]
        n_needed = sentences_per_prof + (1 if i < remainder else 0)

        if len(candidates) >= n_needed:
            chosen = rng.sample(candidates, n_needed)
        else:
            chosen = candidates[:]
            while len(chosen) < n_needed:
                rng.shuffle(candidates)
                chosen.extend(candidates)
            chosen = chosen[:n_needed]

        all_sentences.extend(chosen)

    rng.shuffle(all_sentences)
    return all_sentences


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    rng = random.Random(RANDOM_SEED)

    n_neutral  = sum(1 for t in ALL_PROFESSION_TEMPLATES if isinstance(t, str))
    n_gendered = sum(1 for t in ALL_PROFESSION_TEMPLATES if isinstance(t, tuple))

    # Layer 2 count: anchor_pairs × professions × templates × 2 genders
    n_anchor_prof = (
        len(ANCHOR_PAIRS) * len(PROFESSIONS_TI)
        * len(ALL_ANCHOR_PROF_TEMPLATES) * 2
    )

    print("Tigrigna Corpus Builder")
    print("=" * 60)
    print(f"Professions:                  {len(PROFESSIONS_TI)}")
    print(f"Anchor pairs:                 {len(ANCHOR_PAIRS)}")
    print()
    print(f"Layer 1 — profession templates")
    print(f"  neutral:                    {n_neutral}")
    print(f"  gendered pairs:             {n_gendered}")
    print(f"  target sentences:           {TARGET_PROFESSION_SENTENCES}")
    print()
    print(f"Layer 2 — anchor-profession sentences")
    print(f"  templates:                  {len(ALL_ANCHOR_PROF_TEMPLATES)}")
    print(f"  sentences (pairs×profs×tmpl×2): {n_anchor_prof}")
    print()
    print(f"Total corpus size:            ~{TARGET_PROFESSION_SENTENCES + n_anchor_prof}")

    # Generate both layers
    layer1 = generate_profession_sentences(
        PROFESSIONS_TI, ALL_PROFESSION_TEMPLATES,
        TARGET_PROFESSION_SENTENCES, rng
    )
    layer2 = generate_anchor_profession_sentences(
        ALL_ANCHOR_PROF_TEMPLATES, ANCHOR_PAIRS, PROFESSIONS_TI, rng
    )

    all_sentences = layer1 + layer2
    rng.shuffle(all_sentences)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(all_sentences), encoding="utf-8")

    print(f"\n✓ Wrote {len(all_sentences)} sentences → {OUTPUT_PATH}")
    print(f"  Layer 1 (profession):       {len(layer1)}")
    print(f"  Layer 2 (anchor-profession):{len(layer2)}")

    print("\nSpot-check Layer 2 (first 6 anchor-profession sentences):")
    for s in layer2[:6]:
        print(f"  {s}")

    print("\nNEXT STEPS:")
    print("  1. Native speaker: review anchor-profession template pairs")
    print("  2. Correct verb agreement errors and re-run")
    print("  3. Run the alignment check:")
    print("     python3 diagnostic.py")
    print("  4. Run main.py")


if __name__ == "__main__":
    main()