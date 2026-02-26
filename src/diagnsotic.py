
from transformers import AutoTokenizer
from data_loader import load_professions, load_corpus

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", local_files_only=True)
corpus = load_corpus("ti")
professions = load_professions("ti")

print(f"{'TERM':<25} {'TOKENS':<35} {'CORPUS HITS':>12}  STATUS")
print("-" * 80)
for prof in professions:
    tokens = tokenizer.tokenize(prof)
    # string match (fast approximation)
    hits = sum(1 for s in corpus if prof in s)
    status = "✓" if hits >= 5 else f"✗ only {hits}"
    print(f"{prof:<25} {str(tokens):<35} {hits:>12}  {status}")
