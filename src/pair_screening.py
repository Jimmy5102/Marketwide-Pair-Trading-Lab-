
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
PRICE_PATH = ROOT / "data" / "processed" / "clean_prices.csv"
OUTPUT_DIR = ROOT / "results"

FORMATION_END = "2023-12-29"
FORMATION_DAYS = 252
FDR_ALPHA = 0.05
MAX_HALF_LIFE = 60

MODELS = {
    "baseline": "baseline_candidates.csv",
    "sector": "sector_candidates.csv",
    "cluster": "cluster_candidates.csv",
}


def load_formation_prices():
    if not PRICE_PATH.exists():
        raise FileNotFoundError(f"Missing price file: {PRICE_PATH}")

    prices = pd.read_csv(
        PRICE_PATH, index_col=0, parse_dates=True
    ).sort_index()

    prices = prices.loc[~prices.index.duplicated(keep="last")]
    formation = prices.loc[:FORMATION_END].tail(FORMATION_DAYS)

    if len(formation) != FORMATION_DAYS:
        raise ValueError(
            f"Expected {FORMATION_DAYS} formation dates, "
            f"found {len(formation)}."
        )

    return formation


def test_pair(formation, ticker_a, ticker_b):
    if ticker_a not in formation.columns or ticker_b not in formation.columns:
        return None

    pair = formation[[ticker_a, ticker_b]].dropna()

    if len(pair) != FORMATION_DAYS or (pair <= 0).any().any():
        return None

    log_a = np.log(pair[ticker_a])
    log_b = np.log(pair[ticker_b])

    if log_a.nunique() < 2 or log_b.nunique() < 2:
        return None

    try:
        test_stat, p_value, critical_values = coint(
            log_a, log_b, trend="c", autolag="aic"
        )

        if not np.isfinite(p_value):
            return None

        regression = sm.OLS(
            log_a, sm.add_constant(log_b)
        ).fit()

        intercept = float(regression.params.iloc[0])
        hedge_ratio = float(regression.params.iloc[1])

        spread = log_a - (intercept + hedge_ratio * log_b)

        lagged = spread.shift(1).iloc[1:]
        delta = spread.diff().iloc[1:]

        ar_model = sm.OLS(
            delta, sm.add_constant(lagged)
        ).fit()

        lambda_value = float(ar_model.params.iloc[1])

        half_life = (
            float(-np.log(2) / lambda_value)
            if lambda_value < 0
            else np.inf
        )

        return {
            "ticker_a": ticker_a,
            "ticker_b": ticker_b,
            "observations": len(pair),
            "cointegration_stat": float(test_stat),
            "p_value": float(p_value),
            "critical_1pct": float(critical_values[0]),
            "critical_5pct": float(critical_values[1]),
            "critical_10pct": float(critical_values[2]),
            "intercept": intercept,
            "hedge_ratio": hedge_ratio,
            "spread_std": float(spread.std()),
            "half_life_days": half_life,
        }

    except (ValueError, np.linalg.LinAlgError, ZeroDivisionError):
        return None


def screen_model(model_name, filename, formation):
    candidate_path = OUTPUT_DIR / filename

    if not candidate_path.exists():
        raise FileNotFoundError(
            f"Missing {candidate_path}. Run the corresponding "
            "candidate-generation script first."
        )

    candidates = pd.read_csv(candidate_path)
    results = []

    print(f"\nScreening {model_name}: {len(candidates)} candidates")

    for i, row in candidates.iterrows():
        result = test_pair(
            formation, row["ticker_a"], row["ticker_b"]
        )

        if result is not None:
            result["model"] = model_name
            result["candidate_rank"] = int(row["rank"])
            result["distance"] = float(row["distance"])

            # Retain the generator's metadata where available.
            for column in ["sector", "cluster"]:
                if column in candidates.columns:
                    result[column] = row[column]

            results.append(result)

        if (i + 1) % 100 == 0 or i + 1 == len(candidates):
            print(f"  Processed {i + 1}/{len(candidates)}")

    if not results:
        print(f"  WARNING: No valid tests for {model_name}")
        return pd.DataFrame()

    screened = pd.DataFrame(results)

    # Apply Benjamini-Hochberg within this model's tested family.
    reject, adjusted_p, _, _ = multipletests(
        screened["p_value"].to_numpy(),
        alpha=FDR_ALPHA,
        method="fdr_bh",
    )

    screened["fdr_adjusted_p"] = adjusted_p
    screened["fdr_significant_5pct"] = reject

    screened["passes_basic_filters"] = (
        screened["fdr_significant_5pct"]
        & (screened["hedge_ratio"] > 0)
        & np.isfinite(screened["half_life_days"])
        & (screened["half_life_days"] > 0)
        & (screened["half_life_days"] <= MAX_HALF_LIFE)
    )

    screened = screened.sort_values(
        ["passes_basic_filters", "fdr_adjusted_p", "p_value"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    screened.to_csv(
        OUTPUT_DIR / f"{model_name}_screening_results.csv",
        index=False,
    )

    screened.loc[screened["passes_basic_filters"]].to_csv(
        OUTPUT_DIR / f"{model_name}_screened_pairs.csv",
        index=False,
    )

    return screened


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    formation = load_formation_prices()

    all_results = []
    summary_rows = []

    for model_name, filename in MODELS.items():
        screened = screen_model(model_name, filename, formation)

        if screened.empty:
            summary_rows.append({
                "model": model_name,
                "tested_pairs": 0,
                "fdr_discoveries": 0,
                "passing_pairs": 0,
            })
            continue

        all_results.append(screened)

        summary_rows.append({
            "model": model_name,
            "tested_pairs": len(screened),
            "fdr_discoveries": int(
                screened["fdr_significant_5pct"].sum()
            ),
            "passing_pairs": int(
                screened["passes_basic_filters"].sum()
            ),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        OUTPUT_DIR / "screening_model_comparison.csv",
        index=False,
    )

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(
            OUTPUT_DIR / "all_models_screening_results.csv",
            index=False,
        )

    print("\n--- MODEL SCREENING COMPARISON ---")
    print(summary.to_string(index=False))
    print("\nSaved screening results in results/")


if __name__ == "__main__":
    main()
