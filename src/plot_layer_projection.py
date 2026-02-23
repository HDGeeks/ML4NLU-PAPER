# src/step3_plot_projection_curve.py

"""
step3_plot_projection_curve.py

Intent
------
Plot mean projection per layer.

This is a LIBRARY MODULE now:
- main() should call plot_projection_curve(...) with explicit paths.
- No static paths inside this file.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_projection_curve(
    in_csv: str,
    out_png: str,
    out_mean_csv: str,
    title: str,
    value_col: str = "proj",   # default matches your bias CSV
    layer_col: str = "layer",
) -> None:
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    Path(out_mean_csv).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)

    df[layer_col] = pd.to_numeric(df[layer_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[layer_col, value_col])

    if df.empty:
        raise ValueError(f"No valid rows found in input CSV: {in_csv}")

    agg = (
        df.groupby(layer_col, as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: "mean_projection"})
        .sort_values(layer_col)
    )

    agg.to_csv(out_mean_csv, index=False)

    plt.figure()
    plt.plot(agg[layer_col], agg["mean_projection"], marker="o")
    plt.xlabel("Layer")
    plt.ylabel("Mean projection")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

# """
# step3_plot_projection_curve.py

# Intent
# ------
# Simple plotting script.

# Reads projection CSV.
# Computes mean projection per layer.
# Saves:
#   - PNG curve
#   - layer-mean CSV

# Edit CONFIG block only.
# """

# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path


# # ============================================================
# # ====================== CONFIG BLOCK ========================
# # ============================================================

# IN_CSV = "output/spanish_mbert_projection.csv"
# OUT_PNG = "output/figs/spanish_mbert_projection_curve.png"
# OUT_MEAN_CSV = "output/spanish_mbert_projection_layer_mean.csv"

# TITLE = "Layer-wise mean projection (Spanish | mBERT)"

# # ============================================================


# def main():
#     Path(OUT_PNG).parent.mkdir(parents=True, exist_ok=True)

#     df = pd.read_csv(IN_CSV)

#     df["layer"] = pd.to_numeric(df["layer"], errors="coerce")
#     df["projection"] = pd.to_numeric(df["projection"], errors="coerce")
#     df = df.dropna(subset=["layer", "projection"])

#     if df.empty:
#         raise ValueError("No valid rows found in input CSV.")

#     # mean per layer
#     agg = (
#         df.groupby("layer", as_index=False)["projection"]
#         .mean()
#         .rename(columns={"projection": "mean_projection"})
#         .sort_values("layer")
#     )

#     # save mean CSV
#     agg.to_csv(OUT_MEAN_CSV, index=False)

#     # plot
#     plt.figure()
#     plt.plot(agg["layer"], agg["mean_projection"], marker="o")
#     plt.xlabel("Layer")
#     plt.ylabel("Mean projection")
#     plt.title(TITLE)
#     plt.tight_layout()
#     plt.savefig(OUT_PNG, dpi=200)
#     plt.close()

#     print("Saved plot:", OUT_PNG)
#     print("Saved layer-mean CSV:", OUT_MEAN_CSV)


# if __name__ == "__main__":
#     main()