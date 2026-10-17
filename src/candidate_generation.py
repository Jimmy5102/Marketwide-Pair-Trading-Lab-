
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "clean_prices.csv"
OUTPUT_DIR = ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FORMATION_END = "2023-12-29"
FORMATION_DAYS = 252
MIN_PAIRS_OVERLAP = 252
TOP_N_PAIRS = 500


def load_prices():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Complete Step 2 first."
        )

    prices = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    prices = prices.sort_index()
    prices = prices.loc[~prices.index.duplicated(keep="last")]
    prices = prices.apply(pd.to_numeric, errors="coerce")
    return prices


def build_distance_candidates(prices):
    # Only use observations available on or before the cutoff.
    historical = prices.loc[:FORMATION_END].copy()

    if historical.empty:
        raise ValueError(
            "No historical data exists before FORMATION_END. "
            "Check your dataset's date range."
        )

    # Select the last 252 available trading dates.
    formation = historical.tail(FORMATION_DAYS)

    if len(formation) < FORMATION_DAYS:
        raise ValueError(
            f"Only {len(formation)} formation dates available; "
            f"need {FORMATION_DAYS}."
        )

    # Require complete observations for every selected stock during
    # the common formation window. This makes path distances comparable.
    complete = formation.columns[formation.notna().all(axis=0)]
    formation = formation[complete]

    # Reject non-positive prices and unusable series.
    valid_columns = [
        col for col in formation.columns
        if (formation[col] > 0).all()
        and formation[col].nunique() > 1
    ]
    formation = formation[valid_columns]

    n_stocks = formation.shape[1]

    print("Formation start:", formation.index.min().date())
    print("Formation end:", formation.index.max().date())
    print("Formation observations:", len(formation))
    print("Stocks with complete formation data:", n_stocks)

    if n_stocks < 2:
        raise ValueError("Fewer than two stocks have complete data.")

    # Normalize each stock's price path to 1 at the start.
    normalized = formation.div(formation.iloc[0], axis=1)

    # Distance matrix: sum of squared normalized-price differences.
    values = normalized.to_numpy(dtype=float)
    differences = values[:, :, None] - values[:, None, :]
    distances = np.sum(differences ** 2, axis=0)

    # Generate each unique pair once, excluding the diagonal.
    i_idx, j_idx = np.triu_indices(n_stocks, k=1)

    candidates = pd.DataFrame({
        "ticker_a": normalized.columns.to_numpy()[i_idx],
        "ticker_b": normalized.columns.to_numpy()[j_idx],
        "distance": distances[i_idx, j_idx],
        "overlap_days": len(formation),
    })

    candidates = candidates.sort_values(
        "distance", ascending=True
    ).reset_index(drop=True)

    candidates.insert(
        0, "rank", np.arange(1, len(candidates) + 1)
    )

    return candidates, formation


def main():
    prices = load_prices()

    candidates, formation = build_distance_candidates(prices)

    # Save all scored pairs for auditing and later comparison.
    all_path = OUTPUT_DIR / "baseline_all_pairs.csv"
    candidates.to_csv(all_path, index=False)

    # Save a manageable shortlist for the next research stage.
    shortlist = candidates.head(TOP_N_PAIRS).copy()
    shortlist_path = OUTPUT_DIR / "baseline_candidates.csv"
    shortlist.to_csv(shortlist_path, index=False)

    print("\n--- BASELINE SUMMARY ---")
    print("Stocks used:", formation.shape[1])
    print("All unique pairs:", len(candidates))
    print("Shortlisted pairs:", len(shortlist))
    print("\nTop 20 candidate pairs:")
    print(shortlist.head(20).to_string(index=False))

    print("\nSaved:", all_path)
    print("Saved:", shortlist_path)


if __name__ == "__main__":
    main()
