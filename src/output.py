"""
output.py
---------
Everything that writes to disk after scoring:
  - Tee: mirrors print() to both terminal and a log file simultaneously
  - plot_curve: saves layer-wise mean projection PNG + CSV
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path


class Tee:
    """Write to both stdout and a file simultaneously."""

    def __init__(self, filepath: str):
        self.terminal = sys.stdout
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.logfile  = open(filepath, "w", encoding="utf-8")

    def write(self, msg: str):
        self.terminal.write(msg)
        self.logfile.write(msg)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def close(self):
        sys.stdout = self.terminal
        self.logfile.close()


def plot_curve(all_records: list, out_png: str, out_csv: str, title: str):
    """
    Compute per-layer mean projection, save to CSV, and save a line-plot PNG.
    """
    import matplotlib.pyplot as plt

    by_layer = defaultdict(list)
    for r in all_records:
        by_layer[r["layer"]].append(r["proj"])

    layers = sorted(by_layer.keys())
    means  = [sum(by_layer[l]) / len(by_layer[l]) for l in layers]

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "mean_proj"])
        for l, m in zip(layers, means):
            writer.writerow([l, m])

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(layers, means, marker="o", linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Mean Projection Score\n(+ = male,  - = female)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
