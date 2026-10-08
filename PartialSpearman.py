"""
Partial Spearman rank correlation removes shared monotonic association with
covariates (PSQI and PSS-10) before assessing the relationship between two
variables. Ranks are computed first, then partial correlation is extracted
from the inverse of the rank-correlation matrix — the standard partial
Spearman approach for non-parametric data.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, t as t_dist

PH_COL = "pH"
SWLS_COL = "SWLS Score"
PSQI_COL = "PSQI Score"
PSS_COL = "PSS-10 Score"

ANALYSIS_COLS = [PH_COL, SWLS_COL, PSQI_COL, PSS_COL]
CONTROL_COLS = [PSQI_COL, PSS_COL]

# Pairs where neither variable is a control covariate.
PARTIAL_PAIRS = [
    (PH_COL, SWLS_COL),
]


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce analysis columns to numeric and drop incomplete rows."""
    work = df[ANALYSIS_COLS].copy()
    for col in ANALYSIS_COLS:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    return work.dropna()


def rank_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Average-rank tie handling via scipy.stats.rankdata."""
    ranked = df[columns].copy()
    for col in columns:
        ranked[col] = rankdata(ranked[col].to_numpy(dtype=float), method="average")
    return ranked


def correlation_matrix(values: np.ndarray) -> np.ndarray:
    """Pearson correlation of columns (used on ranks for Spearman)."""
    if values.ndim != 2:
        raise ValueError("Expected a 2D array of shape (n_samples, n_variables).")
    return np.corrcoef(values, rowvar=False)


def partial_correlation_from_corr(corr: np.ndarray, i: int, j: int) -> float:
    """
    Partial correlation rho_ij|rest from full correlation matrix via precision matrix.

    For precision matrix P = corr^{-1}, partial rho_ij = -P_ij / sqrt(P_ii * P_jj).
    """
    precision = np.linalg.pinv(corr)
    denom = np.sqrt(precision[i, i] * precision[j, j])
    if denom == 0:
        return np.nan
    return -precision[i, j] / denom


def partial_spearman_pvalue(rho: float, n: int, n_controls: int) -> float:
    """Two-tailed p-value for partial correlation (t approximation)."""
    df = n - n_controls - 2
    if df < 1 or not np.isfinite(rho) or abs(rho) >= 1:
        return np.nan
    t_stat = rho * np.sqrt(df / (1.0 - rho**2))
    return 2.0 * t_dist.sf(abs(t_stat), df)


def format_corr(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.3f}"


def format_pvalue(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    if value >= 0.001:
        return f"{value:.3f}"
    return f"{value:.3e}"


def pair_label(var_a: str, var_b: str) -> str:
    return f"{var_a} vs {var_b}"


def compute_partial_spearman(
    ranked_df: pd.DataFrame,
    var_a: str,
    var_b: str,
    control_cols: list[str],
) -> dict:
    """Partial Spearman rho and p-value for var_a vs var_b | controls."""
    ordered_cols = [var_a, var_b, *control_cols]
    corr = correlation_matrix(ranked_df[ordered_cols].to_numpy(dtype=float))
    rho = partial_correlation_from_corr(corr, i=0, j=1)
    p_value = partial_spearman_pvalue(rho, n=len(ranked_df), n_controls=len(control_cols))
    return {
        "pair": pair_label(var_a, var_b),
        "var_a": var_a,
        "var_b": var_b,
        "partial_rho": rho,
        "partial_p": p_value,
    }


def compute_bivariate_spearman(clean_df: pd.DataFrame, var_a: str, var_b: str) -> dict:
    rho, p_value = spearmanr(
        clean_df[var_a].to_numpy(dtype=float),
        clean_df[var_b].to_numpy(dtype=float),
    )
    return {
        "pair": pair_label(var_a, var_b),
        "bivariate_rho": rho,
        "bivariate_p": p_value,
    }


def print_results(
    clean_df: pd.DataFrame,
    partial_rows: list[dict],
    bivariate_by_pair: dict[str, dict],
) -> None:
    control_label = ", ".join(CONTROL_COLS)
    print(f"Partial Spearman correlations controlling for {control_label}")
    print(
        f"{'Pair':<34} {'Bivariate rho':>14} {'p (bivariate)':>14} "
        f"{'Partial rho':>12} {'p (partial)':>12} {'Delta rho':>10}"
    )
    print("-" * 98)
    for row in partial_rows:
        bivariate = bivariate_by_pair[row["pair"]]
        delta = row["partial_rho"] - bivariate["bivariate_rho"]
        print(
            f"{row['pair']:<34} "
            f"{format_corr(bivariate['bivariate_rho']):>14} "
            f"{format_pvalue(bivariate['bivariate_p']):>14} "
            f"{format_corr(row['partial_rho']):>12} "
            f"{format_pvalue(row['partial_p']):>12} "
            f"{format_corr(delta):>10}"
        )
    print()
    print(f"N = {len(clean_df)} complete cases")
    print(f"Controls: {control_label} (k = {len(CONTROL_COLS)})")
    print(f"Partial correlation df = N - k - 2 = {len(clean_df) - len(CONTROL_COLS) - 2}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Partial Spearman correlation for salivary pH relationships, "
            "controlling for PSQI and PSS-10."
        )
    )
    parser.add_argument("--csv", default="ShivDataSet.csv", help="Path to CSV file")
    args = parser.parse_args()

    raw_df = pd.read_csv(args.csv)
    clean_df = prepare_df(raw_df)
    min_n = len(CONTROL_COLS) + 3
    if len(clean_df) < min_n:
        raise ValueError(
            f"Not enough complete rows for partial correlation (n={len(clean_df)}; "
            f"need at least {min_n})."
        )

    ranked_df = rank_columns(clean_df, ANALYSIS_COLS)

    partial_rows = [
        compute_partial_spearman(ranked_df, var_a, var_b, CONTROL_COLS)
        for var_a, var_b in PARTIAL_PAIRS
    ]
    bivariate_rows = [
        compute_bivariate_spearman(clean_df, var_a, var_b)
        for var_a, var_b in PARTIAL_PAIRS
    ]
    bivariate_by_pair = {row["pair"]: row for row in bivariate_rows}

    print("Partial Spearman Analysis: pH and SWLS | PSQI + PSS-10")
    print(f"Data: {args.csv}")
    print()
    print_results(clean_df, partial_rows, bivariate_by_pair)


if __name__ == "__main__":
    main()
