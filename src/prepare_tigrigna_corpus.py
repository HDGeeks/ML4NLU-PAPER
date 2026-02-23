import random
from pathlib import Path

RAW = Path("data/tigrigna/corpus_ti_raw.txt")
OUT = Path("data/tigrigna/corpus_ti.txt")

TARGET_SIZE = 8000
MIN_LEN = 20

def main():
    with open(RAW, encoding="utf-8") as f:
        lines = [l.strip() for l in f if len(l.strip()) >= MIN_LEN]

    print("Total usable sentences:", len(lines))

    sample = random.sample(lines, min(TARGET_SIZE, len(lines)))

    with open(OUT, "w", encoding="utf-8") as f:
        for s in sample:
            f.write(s + "\n")

    print("Saved:", len(sample))

if __name__ == "__main__":
    main()