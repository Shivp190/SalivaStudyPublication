"""
Pearson correlation heatmap for salivary pH and three psychological measures
(life satisfaction, sleep quality, perceived stress). Visualizes pairwise linear
associations among all four continuous variables on the same complete-case sample.
"""
import argparse

import matplotlib.pyplot as plt
import pandas as pd

SWLS_COL = "SWLS Score"
PSQI_COL = "PSQI Score"
PSS_COL = "PSS-10 Score"
VARS = ["pH", "SWLS", "PSQI", "PSS-10"]


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce four continuous columns to numeric and drop incomplete rows."""
    work = df.copy()
    work["pH"] = pd.to_numeric(work["pH"], errors="coerce")
    work["SWLS"] = pd.to_numeric(work[SWLS_COL], errors="coerce")
    work["PSQI"] = pd.to_numeric(work[PSQI_COL], errors="coerce")
    work["PSS-10"] = pd.to_numeric(work[PSS_COL], errors="coerce")
    return work[VARS].dropna()


def plot_correlation_heatmap(model_df: pd.DataFrame) -> None:
    """Plot a Pearson correlation heatmap for pH, SWLS, PSQI, and PSS-10."""
    corr = model_df[VARS].corr()

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label="Pearson r")

    labels = list(corr.columns)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Correlation Heatmap: pH, SWLS, PSQI, PSS-10")

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                f"{corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )

    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pearson correlation heatmap for salivary pH and psychological measures."
    )
    parser.add_argument("--csv", default="ShivDataSet.csv", help="Path to CSV file")
    args = parser.parse_args()

    raw_df = pd.read_csv(args.csv)
    model_df = prepare_df(raw_df)

    if len(model_df) < 3:
        raise ValueError(
            f"Not enough complete rows for correlation heatmap (n={len(model_df)})."
        )

    print("Pearson Correlation Heatmap")
    print(f"Data: {args.csv}")
    print(f"N = {len(model_df)}")
    print()

    plot_correlation_heatmap(model_df)


if __name__ == "__main__":
    main()
