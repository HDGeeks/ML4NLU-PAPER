# ── Language & model ──────────────────────────────────────────────────────────
LANGUAGE = "ti"   # "en" | "es" | "ar" | "ti"

# MODEL_NAMES can be a single string OR a list — all will be run in sequence.
MODEL_NAMES = [
    "xlm-roberta-base",
    "xlm-roberta-large",
    "facebook/xlm-v-base",
    "microsoft/mdeberta-v3-base",
]
# Uncomment to run a single model only:
# MODEL_NAMES = "xlm-roberta-base"

# Short slugs used in output filenames and paper figure paths.
MODEL_SLUG_MAP = {
    "bert-base-multilingual-cased": "mbert",
    "xlm-roberta-base":             "xlmr_base",
    "xlm-roberta-large":            "xlmr_large",
    "facebook/xlm-v-base":          "xlmv_base",
    "microsoft/mdeberta-v3-base":   "mdeberta",
}

# ── Run mode ──────────────────────────────────────────────────────────────────
# "debug" → single profession, full verbose trace
# "bulk"  → all professions, one progress line per word
MODE = "bulk"

# ── Debug word (only used when MODE = "debug") ────────────────────────────────
DEBUG_WORD = "ሓረስታይ"

# ── Output root ───────────────────────────────────────────────────────────────
# Change to "output_refactored" to test without overwriting existing results.
# Change back to "output" for normal use.
OUTPUT_ROOT = "output_refactored"

# ── Context counts ────────────────────────────────────────────────────────────
ANCHOR_CONTEXTS     = 12   # sentences per anchor word
PROFESSION_CONTEXTS = 20   # sentences per profession term
