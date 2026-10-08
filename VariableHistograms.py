"""
Histograms of salivary pH and three psychological measures (SWLS, PSQI, PSS-10)
on the same complete-case sample. All four distributions are drawn in one figure.
"""
import argparse

import matplotlib.pyplot as plt
import pandas as pd

SWLS_COL = "SWLS Score"
PSQI_COL = "PSQI Score"
PSS_COL = "PSS-10 Score"
VARS = ["pH", "SWLS", "PSQI", "PSS-10"]
VAR_TITLES = {
    "pH": "Salivary pH",
    "SWLS": "Life Satisfaction (SWLS)",
    "PSQI": "Sleep Quality (PSQI)",
    "PSS-10": "Perceived Stress (PSS-10)",
}
HIST_COLOR = "#5DD9E8"
DEFAULT_OUTPUT = "variable_histograms.png"


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce four continuous columns to numeric and drop incomplete rows."""
    work = df.copy()
    work["pH"] = pd.to_numeric(work["pH"], errors="coerce")
    work["SWLS"] = pd.to_numeric(work[SWLS_COL], errors="coerce")
    work["PSQI"] = pd.to_numeric(work[PSQI_COL], errors="coerce")
    work["PSS-10"] = pd.to_numeric(work[PSS_COL], errors="coerce")
    return work[VARS].dropna()


def plot_variable_histograms(model_df: pd.DataFrame, output_file: str | None = None) -> None:
    """Draw all four variable histograms as a single 2x2 figure."""
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.4))
    axes_flat = axes.ravel()

    for ax, col in zip(axes_flat, VARS):
        ax.hist(
            model_df[col],
            bins="auto",
            color=HIST_COLOR,
            edgecolor="white",
            linewidth=0.8,
        )
        ax.set_title(VAR_TITLES[col])
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")

    fig.suptitle("Distributions of pH, SWLS, PSQI, and PSS-10", fontsize=13)
    fig.tight_layout()

    if output_file:
        fig.savefig(output_file, dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Histograms of salivary pH and psychological measures in one image."
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
            f"Not enough complete rows for histograms (n={len(model_df)})."
        )

    print("Variable Histograms")
    print(f"Data: {args.csv}")
    print(f"N = {len(model_df)}")
    print(f"Output: {args.output}")
    print()

    plot_variable_histograms(model_df, output_file=args.output)


if __name__ == "__main__":
    main()
