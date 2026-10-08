"""
Scatterplot of salivary pH (y) against life satisfaction (SWLS, x)
on the same complete-case sample used in the other figures.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SWLS_COL = "SWLS Score"
POINT_COLOR = "#5DD9E8"
LINE_COLOR = "black"
DEFAULT_OUTPUT = "ph_swls_scatter.png"


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce pH and SWLS to numeric and drop incomplete rows."""
    work = df.copy()
    work["pH"] = pd.to_numeric(work["pH"], errors="coerce")
    work["SWLS"] = pd.to_numeric(work[SWLS_COL], errors="coerce")
    return work[["pH", "SWLS"]].dropna()


def plot_ph_swls_scatter(model_df: pd.DataFrame, output_file: str | None = None) -> None:
    """Draw pH vs SWLS scatterplot with a linear trend line."""
    x = model_df["SWLS"].to_numpy(dtype=float)
    y = model_df["pH"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.scatter(x, y, color=POINT_COLOR, edgecolors="black", linewidths=0.7, s=42, zorder=3)

    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color=LINE_COLOR, linewidth=1.6, zorder=2)

    ax.set_xlabel("Life Satisfaction (SWLS)")
    ax.set_ylabel("Salivary pH")
    ax.set_title("Salivary pH vs Life Satisfaction (SWLS)")
    fig.tight_layout()

    if output_file:
        fig.savefig(output_file, dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scatterplot of salivary pH (y) against SWLS (x)."
    )
    parser.add_argument("--csv", default="ShivDataSet.csv", help="Path to CSV file")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output image path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    raw_df = pd.read_csv(args.csv)
    model_df = prepare_df(raw_df)

    if len(model_df) < 3:
        raise ValueError(
            f"Not enough complete rows for scatterplot (n={len(model_df)})."
        )

    print("pH vs SWLS Scatterplot")
    print(f"Data: {args.csv}")
    print(f"N = {len(model_df)}")
    print(f"Output: {args.output}")
    print()

    plot_ph_swls_scatter(model_df, output_file=args.output)


if __name__ == "__main__":
    main()
