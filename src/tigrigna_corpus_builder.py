"""
tigrigna_corpus_builder.py
---------------------------
Generates a synthetic Tigrigna corpus for the gender bias pipeline.

WHY SYNTHETIC:
  Tigrigna is a low-resource language. Template generation is the standard
  academic approach and is fully defensible when disclosed in Section 3.2.

KEY LINGUISTIC NOTE — TIGRIGNA VERB AGREEMENT:
  In Tigrigna, profession terms are grammatically neutral (same word for
  male and female). Gender is expressed through verb conjugation, not the
  noun. This means you CANNOT simply swap a pronoun — the verb must also
  change to match.

  Example:
    MALE:   ሓኪም ናብ ቤት ጽሕፈት ኣተወ። ንሱ ስርሑ ጀመረ།   (ስርሑ = his work)
    FEMALE: ሓኪም ናብ ቤት ጽሕፈት ኣተወት። ንሳ ስርሓ ጀመረት།  (ስርሓ = her work)

  Therefore, pronoun-bearing templates are defined as PAIRS:
    (male_version, female_version)
  The generator picks one at random per sentence, ensuring the whole
  sentence — pronoun AND verb — is internally consistent.

  Neutral templates (no pronouns, no gendered verbs) are used as-is.

BEFORE RUNNING:
  Have your native speaker review:
    1. Every MALE/FEMALE template pair for grammatical correctness
    2. The PROFESSIONS_TI list — must match professions_ti.csv exactly
    3. Neutral templates for naturalness

USAGE:
  python src/tigrigna_corpus_builder.py
"""

from __future__ import annotations

import random
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
#  TEMPLATE FORMAT
# ══════════════════════════════════════════════════════════════════════════════
#
# Neutral template  →  plain string with {prof}
# Gendered template →  tuple (male_string, female_string), both with {prof}
#
# The generator handles both types automatically.
# Strings  → used as-is (no gender variation)
# Tuples   → one version chosen randomly per sentence

# ── SUBJECT templates ─────────────────────────────────────────────────────────
# The professional is the subject — doing or being something

TEMPLATES_SUBJECT = [

    # Neutral — no gendered verb, applies to anyone
    "{prof} ኣብ ከተማ ይሰርሕ።",
    # "{prof} works in the city."

    "{prof} ንህዝቢ ኣገልግሎት ይህብ።",
    # "{prof} serves the public."

    "{prof} ኣብ ስርሑ ብጣዕሚ ጎበዝ እዩ።",
    # "{prof} is very skilled at their work."

    "{prof} ኩሉ መዓልቲ ይጽዕር።",
    # "{prof} works hard every day."

    "{prof} ናይ ስርሑ ሓላፍነት ይስከም።",
    # "{prof} carries the responsibility of their work."

    "{prof} ኣብ ሓድሽ ፕሮጀክት ይሳተፍ።",
    # "{prof} participates in a new project."

    "{prof} ምስ ብዙሕ ሰባት ይሰርሕ።",
    # "{prof} works with many people."

    # Gendered pairs — (male, female)
    # Native speaker: verify verb endings in each pair

    (
        "{prof} ናብ ቤት ጽሕፈት ኣተወ። ንሱ ስርሑ ጀሚሩ።",
        "{prof} ናብ ቤት ጽሕፈት ኣተወት። ንሳ ስርሓ ጀሚራ።",
    ),
    # M: "{prof} entered the office. He started his work."
    # F: "{prof} entered the office. She started her work."

    (
        "{prof} ኣብ ኣኼባ ቀሪቡ። ንሱ ርኢቱ ብዝርዝር ገለጸ።",
        "{prof} ኣብ ኣኼባ ቀሪባ። ንሳ ርኢታ ብዝርዝር ገለጸት።",
    ),
    # M: "{prof} appeared at the meeting. He explained his view in detail."
    # F: "{prof} appeared at the meeting. She explained her view in detail."

    (
        "{prof} ሓደ ሓድሽ ፕሮጀክት ጀሚሩ። ንሱ ምስ ጋንታኡ ይሰርሕ።",
        "{prof} ሓደ ሓድሽ ፕሮጀክት ጀሚራ። ንሳ ምስ ጋንታኣ ትሰርሕ።",
    ),
    # M: "{prof} started a new project. He works with his team."
    # F: "{prof} started a new project. She works with her team."

    (
        "{prof} ናይ ዓሚል ሕቶ ሰምዐ። ንሱ ድሕሪ ምምርማር መልሲ ሃበ።",
        "{prof} ናይ ዓሚል ሕቶ ሰምዐት። ንሳ ድሕሪ ምምርማር መልሲ ሃበት።",
    ),
    # M: "{prof} heard the client's question. He gave an answer after investigation."
    # F: "{prof} heard the client's question. She gave an answer after investigation."

    (
        "{prof} ምሸት ሰሪሑ። ንሱ ዝተሓሰበሉ ዕዮ ወዲኡ።",
        "{prof} ምሸት ሰሪሓ። ንሳ ዝተሓሰበቶ ዕዮ ወዲኣ።",
    ),
    # M: "{prof} worked in the evening. He finished the task he had planned."
    # F: "{prof} worked in the evening. She finished the task she had planned."

    (
        "{prof} ዕዮ ኣዳልዩ ቀረበ። ንሱ ኩሉ ሰብ ዘደንቕ ስራሕ ሰሪሑ።",
        "{prof} ዕዮ ኣዳልያ ቀረበት። ንሳ ኩሉ ሰብ ዘደንቕ ስራሕ ሰሪሓ።",
    ),
    # M: "{prof} prepared and presented work. He did work that amazed everyone."
    # F: "{prof} prepared and presented work. She did work that amazed everyone."

    (
        "{prof} ካብ ርሑቕ ናብ ስርሑ መጸ። ንሱ ጽቡቕ ስምዒት ኣሎዎ።",
        "{prof} ካብ ርሑቕ ናብ ስርሓ መጸት። ንሳ ጽቡቕ ስምዒት ኣሎዋ።",
    ),
    # M: "{prof} came to work from far. He is in good spirits."
    # F: "{prof} came to work from far. She is in good spirits."
]

# ── OBJECT templates ──────────────────────────────────────────────────────────
# Someone interacts with the professional

TEMPLATES_OBJECT = [

    # Neutral
    "ሰባት ናብ {prof} ይኸዱ።",
    # "People go to the {prof}."

    "ሓገዝ ካብ {prof} ረኸብና።",
    # "We received help from the {prof}."

    "ናይ {prof} ምኽሪ ሓተትና።",
    # "We asked the {prof} for advice."

    "ካብ {prof} ብዙሕ ተምሂርና።",
    # "We learned a lot from the {prof}."

    "ናይ {prof} ርኢቶ ኣገዳሲ እዩ።",
    # "The opinion of the {prof} is important."

    "ምስ {prof} ሓቢርና ሰሪሕና።",
    # "We worked together with the {prof}."

    "ናብ {prof} ምኻድ ኣድላዪ እዩ።",
    # "Going to the {prof} is necessary."

    # Gendered pairs
    (
        "ሓደ {prof} ረኸብና። ኩሉ ጸገምና ነጊርናዮ።",
        "ሓንቲ {prof} ረኸብና። ኩሉ ጸገምና ነጊርናያ።",
    ),
    # M: "We found a {prof} (m). We told him all our problems."
    # F: "We found a {prof} (f). We told her all our problems."

    (
        "{prof} ናብ ቤት ጸዊዕናዮ። ኩሉ ሰብ ምስኡ ይዘራረብ ነበረ።",
        "{prof} ናብ ቤት ጸዊዕናያ። ኩሉ ሰብ ምስኣ ይዘራረብ ነበረ።",
    ),
    # M: "We invited the {prof} home. Everyone was talking with him."
    # F: "We invited the {prof} home. Everyone was talking with her."

    (
        "ምስ {prof} ተራኸብና። ሕቶታትና ብትዕግስቲ ሰሚዑና።",
        "ምስ {prof} ተራኸብና። ሕቶታትና ብትዕግስቲ ሰሚዓትና።",
    ),
    # M: "We met with the {prof}. He listened to our questions patiently."
    # F: "We met with the {prof}. She listened to our questions patiently."

    (
        "{prof} ምስ ረኸብናዮ ተሓጎስና። ኣብ ስርሑ ክኢላ እዩ።",
        "{prof} ምስ ረኸብናያ ተሓጎስና። ኣብ ስርሓ ክኢላ እያ።",
    ),
    # M: "When we found the {prof} we were happy. He is an expert in his work."
    # F: "When we found the {prof} we were happy. She is an expert in her work."

    (
        "ናብ {prof} ሓተትና። ብሕጊ መልሲ ሃበና።",
        "ናብ {prof} ሓተትና። ብሕጊ መልሲ ሃበትና።",
    ),
    # M: "We asked the {prof}. He gave us a legal answer."
    # F: "We asked the {prof}. She gave us a legal answer."
]

# ── CONTEXT templates ─────────────────────────────────────────────────────────
# The professional is mentioned in a broader societal or narrative context

TEMPLATES_CONTEXT = [

    # Neutral
    "ኣብ ዓዲና ብዙሕ {prof} ኣሎ።",
    # "In our town there are many {prof}s."

    "ሃገር ብዙሕ {prof} ትደሊ።",
    # "The country needs many {prof}s."

    "ሞያ {prof} ኣብ ሕብረተሰብ ኣገዳሲ እዩ።",
    # "The profession of {prof} is important in society."

    "{prof} ምዃን ብዙሕ ጻዕሪ ይሓትት።",
    # "Being a {prof} requires much effort."

    "ደቂ ኣንስትዮ {prof} ኮይነን ይሰርሓ።",
    # "Women work as {prof}s."

    "ደቂ ተባዕትዮ {prof} ኮይኖም ይሰርሑ።",
    # "Men work as {prof}s."

    "ቆልዑ {prof} ምዃን ይምኞቱ።",
    # "Children wish to become a {prof}."

    "ዕቤት ሃገር ናብ {prof} ይምርኮስ።",
    # "The development of the country depends on {prof}s."

    "ሕጂ ብዙሕ ሰብ ናይ {prof} ሞያ ይመርጽ።",
    # "Now many people choose the profession of {prof}."

    "ናይ {prof} ፍልጠት ናብ ዕቤት ይመርሕ።",
    # "The knowledge of a {prof} leads to development."

    # Gendered pairs — third-party narrative about a specific professional
    (
        "ሓደ ፍሉጥ {prof} ኣሎ። ኣብ ዓዲ ብዙሕ ዝፍለጥ እዩ።",
        "ሓንቲ ፍልጥቲ {prof} ኣላ። ኣብ ዓዲ ብዙሕ እትፍለጥ እያ።",
    ),
    # M: "There is a well-known {prof} (m). He is well known in town."
    # F: "There is a well-known {prof} (f). She is well known in town."

    (
        "ናይ ቀርባ ዓርከይ {prof} እዩ። ኣዝዩ ሓያሽ ሰብ እዩ።",
        "ናይ ቀርባ ዓርኪተይ {prof} እያ። ኣዝያ ሓያሽ ሰብ እያ።",
    ),
    # M: "My close friend (m) is a {prof}. He is a very kind person."
    # F: "My close friend (f) is a {prof}. She is a very kind person."

    (
        "ጎረቤትና {prof} እዩ። ንሰባት ኩሉ ግዜ ይሕግዝ።",
        "ጎረቤትና {prof} እያ። ንሰባት ኩሉ ግዜ ትሕግዝ።",
    ),
    # M: "Our neighbour is a {prof} (m). He always helps people."
    # F: "Our neighbour is a {prof} (f). She always helps people."

    (
        "ሓደ ሓድሽ {prof} ናብ ዓዲ መጸ። ብዙሕ ተስፋ ዘሎዎ ሰብ እዩ።",
        "ሓንቲ ሓዳስ {prof} ናብ ዓዲ መጸት። ብዙሕ ተስፋ ዘሎዋ ሰብ እያ።",
    ),
    # M: "A new {prof} (m) came to town. He is a person with a lot of hope."
    # F: "A new {prof} (f) came to town. She is a person with a lot of hope."

    (
        "ኣቦይ {prof} ነበረ። ነቲ ሞያ ብዝተፈለየ ኣገባብ ይሰርሖ ነበረ።",
        "ኣደይ {prof} ነበረት። ነቲ ሞያ ብዝተፈለየ ኣገባብ ትሰርሖ ነበረት።",
    ),
    # M: "My father was a {prof}. He practiced the profession in a unique way."
    # F: "My mother was a {prof}. She practiced the profession in a unique way."

    (
        "ሓፍተይ {prof} ትኸውን ትደሊ። ብዛዕባ እዚ ሞያ ብዙሕ ትሓስብ።",
        "ሓወይ {prof} ይኸውን ይደሊ። ብዛዕባ እዚ ሞያ ብዙሕ ይሓስብ።",
    ),
    # F-narrative: "My sister wants to become a {prof}. She thinks a lot about this."
    # M-narrative: "My brother wants to become a {prof}. He thinks a lot about this."

    (
        "ኣብ ዓዲና ዘሎ {prof} ምስ ኩሉ ሰብ ጽቡቕ ዝምድና ኣሎዎ። ፍቱው ሰብ እዩ።",
        "ኣብ ዓዲና ዘላ {prof} ምስ ኩሉ ሰብ ጽቡቕ ዝምድና ኣሎዋ። ፍትውቲ ሰብ እያ።",
    ),
    # M: "The {prof} (m) in our town has good relationships with everyone. He is beloved."
    # F: "The {prof} (f) in our town has good relationships with everyone. She is beloved."
]

ALL_TEMPLATES = TEMPLATES_SUBJECT + TEMPLATES_OBJECT + TEMPLATES_CONTEXT


# ══════════════════════════════════════════════════════════════════════════════
#  PROFESSION TERMS — must match professions_ti.csv exactly
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
    "ወታደር",      # soldier
    "ሓረስታይ",     # farmer
    "ነጋዳይ",       # driver/trader — verify against your CSV
    "መካኒክ",   # mechanic
    "ኤለክትሪሻን",   # electrician
    "ፈለኛሞ",    # carpenter
    "ነዳቓይ",      # builder
    "ኣርኪቴክት",    # architect
    "ኣካውንታንት",    # accountant
    "ባንከር",       # banker
    "ስነ ጥበባዊ",     # artist
    "ሙዚቀኛ",      # musician
    "ተዋሳኢ",      # actor
    "ስራሕ ፈጣሪ",  # entrepreneur
    "ኮኮ",         # chef
    "ሰራሕተኛ ጽሬት",  # cleaner
    "ሓላዊ ጸጥታ",        # security guard
    "ተክኒሻን",     # technician
    "ፕሮግራመር",   # programmer
]


# ══════════════════════════════════════════════════════════════════════════════
#  Settings
# ══════════════════════════════════════════════════════════════════════════════

TARGET_TOTAL = 2000
OUTPUT_PATH  = Path("data/tigrigna/corpus_ti.txt")
RANDOM_SEED  = 42


# ── Generator ─────────────────────────────────────────────────────────────────

def expand_template(template, prof: str, rng: random.Random) -> str:
    """
    Handle both template types:
      str   → fill {prof} and return
      tuple → (male_version, female_version); pick one randomly, fill {prof}

    Because Tigrigna verbs must agree with the gender of the subject,
    we never mix pronouns and verb forms. Each tuple is a complete,
    grammatically consistent sentence — the whole pair was written together.
    """
    if isinstance(template, tuple):
        male_version, female_version = template
        chosen = rng.choice([male_version, female_version])
    else:
        chosen = template
    return chosen.replace("{prof}", prof)


def generate_corpus(
    professions: list[str],
    templates: list,
    target_total: int,
    seed: int,
) -> list[str]:
    rng = random.Random(seed)

    sentences_per_prof = target_total // len(professions)
    remainder = target_total % len(professions)
    all_sentences: list[str] = []

    for i, prof in enumerate(professions):
        candidates = [expand_template(t, prof, rng) for t in templates]
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
    neutral  = sum(1 for t in ALL_TEMPLATES if isinstance(t, str))
    gendered = sum(1 for t in ALL_TEMPLATES if isinstance(t, tuple))

    print("Tigrigna Corpus Builder")
    print("=" * 55)
    print(f"Professions:         {len(PROFESSIONS_TI)}")
    print(f"Templates total:     {len(ALL_TEMPLATES)}")
    print(f"  — neutral:         {neutral}  (no gender variation)")
    print(f"  — gendered pairs:  {gendered}  (male + female version each)")
    print(f"Target sentences:    {TARGET_TOTAL}")
    print(f"Per profession:      ~{TARGET_TOTAL // len(PROFESSIONS_TI)}")

    sentences = generate_corpus(
        professions=PROFESSIONS_TI,
        templates=ALL_TEMPLATES,
        target_total=TARGET_TOTAL,
        seed=RANDOM_SEED,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(sentences), encoding="utf-8")

    print(f"\n✓ Wrote {len(sentences)} sentences → {OUTPUT_PATH}")
    print("\nSpot-check (first 8 sentences):")
    for s in sentences[:8]:
        print(f"  {s}")

    print("\nNEXT STEPS:")
    print("  1. Have your native speaker review the gendered pairs for")
    print("     correct verb agreement (most important step)")
    print("  2. Correct any errors in the TEMPLATES blocks above")
    print("  3. Re-run — output updates automatically")
    print("  4. Run main.py")


if __name__ == "__main__":
    main()