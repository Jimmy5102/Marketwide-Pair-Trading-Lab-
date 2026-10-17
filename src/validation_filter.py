
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

PRICE_PATH = ROOT / "data" / "processed" / "clean_prices.csv"
RESULTS_DIR = ROOT / "results"

FORMATION_END = "2023-12-29"
FORMATION_DAYS = 252

VALIDATION_START = "2024-01-02"
VALIDATION_END = "2024-02-15"
TEST_START = "2024-02-16"

# Predeclared validation criteria.
MAX_ABS_Z = 4.0
MAX_FINAL_ABS_Z = 2.5
MAX_VOLATILITY_RATIO = 2.0
MIN_VALIDATION_DAYS = 20

MODEL_FILES = {
    "baseline": "baseline_screened_pairs.csv",
    "sector": "sector_screened_pairs.csv",
    "cluster": "cluster_screened_pairs.csv",
}


# --------------------------------------------------
# Load prices
# --------------------------------------------------

def load_prices():
    if not PRICE_PATH.exists():
        raise FileNotFoundError(
            f"Missing price file: {PRICE_PATH}"
        )

    prices = pd.read_csv(
        PRICE_PATH,
        index_col=0,
        parse_dates=True,
    ).sort_index()

    prices = prices.loc[
        ~prices.index.duplicated(keep="last")
    ]

    prices = prices.apply(pd.to_numeric, errors="coerce")

    return prices


# --------------------------------------------------
# Evaluate one pair using formation and validation data
# --------------------------------------------------

def evaluate_pair(prices, row):
    ticker_a = row["ticker_a"]
    ticker_b = row["ticker_b"]

    intercept = float(row["intercept"])
    beta = float(row["hedge_ratio"])

    if not np.isfinite(intercept) or not np.isfinite(beta):
        raise ValueError("Non-finite formation parameters.")

    if beta <= 0:
        raise ValueError("Hedge ratio must be positive.")

    if ticker_a not in prices.columns or ticker_b not in prices.columns:
        raise ValueError("Ticker missing from price data.")

    pair = prices[[ticker_a, ticker_b]].dropna()

    if (pair <= 0).any().any():
        raise ValueError("Prices must be positive.")

    formation = pair.loc[:FORMATION_END].tail(FORMATION_DAYS)

    if len(formation) != FORMATION_DAYS:
        raise ValueError(
            f"Expected {FORMATION_DAYS} formation observations; "
            f"found {len(formation)}."
        )

    formation_spread = (
        np.log(formation[ticker_a])
        - intercept
        - beta * np.log(formation[ticker_b])
    )

    formation_std = float(formation_spread.std(ddof=1))

    if not np.isfinite(formation_std) or formation_std <= 0:
        raise ValueError("Invalid formation spread volatility.")

    # Validation uses fixed formation-period parameters.
    validation = pair.loc[
        (pair.index >= pd.Timestamp(VALIDATION_START))
        & (pair.index <= pd.Timestamp(VALIDATION_END))
    ]

    if len(validation) < MIN_VALIDATION_DAYS:
        raise ValueError(
            f"Only {len(validation)} validation observations."
        )

    validation_spread = (
        np.log(validation[ticker_a])
        - intercept
        - beta * np.log(validation[ticker_b])
    )

    validation_z = (
        validation_spread - float(formation_spread.mean())
    ) / formation_std

    abs_z = validation_z.abs()

    max_abs_z = float(abs_z.max())
    final_abs_z = float(abs_z.iloc[-1])

    validation_std = float(validation_spread.std(ddof=1))
    volatility_ratio = validation_std / formation_std

    if not np.isfinite(volatility_ratio):
        volatility_ratio = np.inf

    days_beyond_3 = int((abs_z > 3.0).sum())
    days_beyond_4 = int((abs_z >= MAX_ABS_Z).sum())

    passes_filter = (
        max_abs_z < MAX_ABS_Z
        and final_abs_z <= MAX_FINAL_ABS_Z
        and volatility_ratio <= MAX_VOLATILITY_RATIO
        and len(validation) >= MIN_VALIDATION_DAYS
    )

    return {
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "validation_days": len(validation),
        "validation_max_abs_z": max_abs_z,
        "validation_final_abs_z": final_abs_z,
        "validation_volatility_ratio": volatility_ratio,
        "days_abs_z_gt_3": days_beyond_3,
        "days_abs_z_ge_4": days_beyond_4,
        "passes_stability_filter": passes_filter,
    }


# --------------------------------------------------
# Run all models
# --------------------------------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_prices()

    summaries = []

    for model_name, filename in MODEL_FILES.items():
        candidate_path = RESULTS_DIR / filename

        print(f"\n{'=' * 60}")
        print(f"MODEL: {model_name.upper()}")

        if not candidate_path.exists():
            print(f"Missing screened-pair file: {candidate_path}")
            print("Skipping this model.")
            summaries.append({
                "model": model_name,
                "input_pairs": 0,
                "valid_pairs": 0,
                "stable_pairs": 0,
            })
            continue

        candidates = pd.read_csv(candidate_path)

        if candidates.empty:
            print("No screened pairs available.")
            # Save an empty file with the expected columns.
            candidates.to_csv(
                RESULTS_DIR / f"{model_name}_stable_pairs.csv",
                index=False,
            )
            summaries.append({
                "model": model_name,
                "input_pairs": 0,
                "valid_pairs": 0,
                "stable_pairs": 0,
            })
            continue

        required = {
            "ticker_a",
            "ticker_b",
            "intercept",
            "hedge_ratio",
        }

        missing = required - set(candidates.columns)

        if missing:
            raise ValueError(
                f"{filename} is missing columns: {sorted(missing)}"
            )

        diagnostics = []
        valid_indices = []

        for index, row in candidates.iterrows():
            try:
                result = evaluate_pair(prices, row)
                diagnostics.append(result)
                valid_indices.append(index)

            except (ValueError, KeyError, IndexError) as exc:
                print(
                    f"Skipping {row.get('ticker_a', '?')}-"
                    f"{row.get('ticker_b', '?')}: {exc}"
                )

        diagnostic_df = pd.DataFrame(diagnostics)

        diagnostic_path = (
            RESULTS_DIR / f"{model_name}_validation_diagnostics.csv"
        )

        diagnostic_df.to_csv(diagnostic_path, index=False)

        if diagnostic_df.empty:
            stable_pairs = candidates.iloc[0:0].copy()
            stable_count = 0
        else:
            passing = diagnostic_df[
                "passes_stability_filter"
            ].to_numpy()

            valid_candidates = candidates.loc[
                valid_indices
            ].reset_index(drop=True)

            stable_pairs = valid_candidates.loc[
                passing
            ].copy()

            stable_count = len(stable_pairs)

        stable_path = RESULTS_DIR / f"{model_name}_stable_pairs.csv"
        stable_pairs.to_csv(stable_path, index=False)

        summary = {
            "model": model_name,
            "input_pairs": len(candidates),
            "valid_pairs": len(diagnostic_df),
            "stable_pairs": stable_count,
        }

        summaries.append(summary)

        print(diagnostic_df.to_string(index=False))
        print(f"\nDiagnostics saved: {diagnostic_path}")
        print(f"Stable pairs saved: {stable_path}")
        print(f"Pairs passing filter: {stable_count}")

    summary_df = pd.DataFrame(summaries)
    summary_path = RESULTS_DIR / "validation_filter_comparison.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n--- VALIDATION FILTER COMPARISON ---")
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()