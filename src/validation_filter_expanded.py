from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PRICE_PATH = ROOT / "data" / "processed" / "clean_russell1000_prices.csv"
PAIRS_PATH = ROOT / "results" / "russell1000_baseline_screened_pairs.csv"
OUTPUT_DIR = ROOT / "results"

FORMATION_END = "2023-12-29"
FORMATION_DAYS = 252
VALIDATION_START = "2024-01-02"
VALIDATION_END = "2024-02-15"

MIN_VALIDATION_DAYS = 20
MAX_ABS_Z = 4.0
MAX_FINAL_ABS_Z = 2.5
MAX_VOLATILITY_RATIO = 2.0


def main():
    prices = pd.read_csv(
        PRICE_PATH, index_col=0, parse_dates=True
    ).sort_index()

    prices = prices.loc[~prices.index.duplicated(keep="last")]
    prices = prices.apply(pd.to_numeric, errors="coerce")

    pairs = pd.read_csv(PAIRS_PATH)

    formation = prices.loc[:FORMATION_END].tail(FORMATION_DAYS)
    validation = prices.loc[VALIDATION_START:VALIDATION_END]

    if len(formation) != FORMATION_DAYS:
        raise ValueError(
            f"Expected {FORMATION_DAYS} formation days; got {len(formation)}."
        )

    if len(validation) < MIN_VALIDATION_DAYS:
        raise ValueError(
            f"Only {len(validation)} validation days; "
            f"need at least {MIN_VALIDATION_DAYS}."
        )

    print("Formation observations:", len(formation))
    print("Validation observations:", len(validation))
    print("Pairs to validate:", len(pairs))

    results = []

    for _, row in pairs.iterrows():
        a = row["ticker_a"]
        b = row["ticker_b"]

        if a not in prices.columns or b not in prices.columns:
            continue

        train_pair = formation[[a, b]].dropna()
        valid_pair = validation[[a, b]].dropna()

        if len(train_pair) != FORMATION_DAYS:
            continue

        if len(valid_pair) < MIN_VALIDATION_DAYS:
            continue

        if (train_pair <= 0).any().any() or (valid_pair <= 0).any().any():
            continue

        log_a = np.log(train_pair[a])
        log_b = np.log(train_pair[b])

        # Reuse formation-period regression parameters only.
        # Do not refit the hedge ratio on validation data.
        beta = float(row["hedge_ratio"])
        intercept = float(row["intercept"])

        train_spread = log_a - (intercept + beta * log_b)
        train_mean = float(train_spread.mean())
        train_std = float(train_spread.std(ddof=1))

        if not np.isfinite(train_std) or train_std <= 0:
            continue

        valid_spread = (
            np.log(valid_pair[a])
            - (intercept + beta * np.log(valid_pair[b]))
        )

        z = (valid_spread - train_mean) / train_std

        max_abs_z = float(z.abs().max())
        final_abs_z = float(abs(z.iloc[-1]))

        validation_volatility = float(valid_spread.std(ddof=1))
        volatility_ratio = validation_volatility / train_std

        stable = (
            max_abs_z < MAX_ABS_Z
            and final_abs_z <= MAX_FINAL_ABS_Z
            and volatility_ratio <= MAX_VOLATILITY_RATIO
        )

        results.append({
            "ticker_a": a,
            "ticker_b": b,
            "candidate_rank": row.get("candidate_rank", np.nan),
            "formation_p_value": row["p_value"],
            "formation_fdr_adjusted_p": row["fdr_adjusted_p"],
            "hedge_ratio": beta,
            "intercept": intercept,
            "formation_spread_mean": train_mean,
            "formation_spread_std": train_std,
            "validation_days": len(valid_pair),
            "validation_max_abs_z": max_abs_z,
            "validation_final_abs_z": final_abs_z,
            "validation_volatility_ratio": volatility_ratio,
            "passes_stability_filter": stable,
        })

    if not results:
        raise RuntimeError("No pairs could be validated.")

    result_df = pd.DataFrame(results).sort_values(
        ["passes_stability_filter", "formation_fdr_adjusted_p"],
        ascending=[False, True],
    ).reset_index(drop=True)

    all_path = OUTPUT_DIR / "russell1000_baseline_validation_results.csv"
    stable_path = OUTPUT_DIR / "russell1000_baseline_stable_pairs.csv"
    summary_path = OUTPUT_DIR / "russell1000_baseline_validation_summary.csv"

    result_df.to_csv(all_path, index=False)
    result_df.loc[result_df["passes_stability_filter"]].to_csv(
        stable_path, index=False
    )

    summary = pd.DataFrame([{
        "model": "russell1000_baseline",
        "screened_pairs": len(pairs),
        "validated_pairs": len(result_df),
        "stable_pairs": int(result_df["passes_stability_filter"].sum()),
        "validation_start": VALIDATION_START,
        "validation_end": VALIDATION_END,
        "max_abs_z_threshold": MAX_ABS_Z,
        "max_final_abs_z_threshold": MAX_FINAL_ABS_Z,
        "max_volatility_ratio": MAX_VOLATILITY_RATIO,
    }])
    summary.to_csv(summary_path, index=False)

    print("\n--- RUSSELL 1000 VALIDATION SUMMARY ---")
    print(summary.to_string(index=False))

    print("\nAll pair diagnostics:")
    print(result_df.to_string(index=False))

    print("\nSaved:", all_path)
    print("Saved:", stable_path)
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()
