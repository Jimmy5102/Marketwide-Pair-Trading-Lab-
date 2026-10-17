from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT / "data" / "processed" /
    "clean_russell1000_prices.csv"
)
OUTPUT_DIR = ROOT / "results"

FORMATION_END = "2023-12-29"
FORMATION_DAYS = 252
TOP_N_PAIRS = 500


def load_prices():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing price file: {DATA_PATH}")

    prices = pd.read_csv(
        DATA_PATH,
        index_col=0,
        parse_dates=True,
    )

    prices = prices.sort_index()
    prices = prices.loc[
        ~prices.index.duplicated(keep="last")
    ]
    prices = prices.apply(pd.to_numeric, errors="coerce")

    # Prevent ambiguous ticker columns from entering the analysis.
    prices = prices.loc[:, ~prices.columns.duplicated()]

    return prices


def build_distance_candidates(prices):
    historical = prices.loc[:FORMATION_END]
    formation = historical.tail(FORMATION_DAYS).copy()

    if len(formation) != FORMATION_DAYS:
        raise ValueError(
            f"Expected {FORMATION_DAYS} formation dates, "
            f"found {len(formation)}."
        )

    # Keep only stocks with complete data during the formation window.
    complete = formation.columns[
        formation.notna().all(axis=0)
    ]
    formation = formation[complete]

    # Require strictly positive prices and nonconstant series.
    valid = [
        ticker
        for ticker in formation.columns
        if (formation[ticker] > 0).all()
        and formation[ticker].nunique() > 1
    ]
    formation = formation[valid]

    if formation.shape[1] < 2:
        raise ValueError(
            "Fewer than two stocks have valid formation data."
        )

    print("Formation start:", formation.index.min().date())
    print("Formation end:", formation.index.max().date())
    print("Formation observations:", len(formation))
    print("Stocks with complete formation data:", formation.shape[1])

    # Normalize each price path to 1 on the first formation date.
    normalized = formation.div(formation.iloc[0], axis=1)

    # Each stock is one row. pdist computes each unique pair once,
    # avoiding the large 3D array used by the original implementation.
    stock_paths = normalized.to_numpy(dtype=float).T
    distances = pdist(stock_paths, metric="sqeuclidean")

    tickers = normalized.columns.to_numpy()
    pairs = list(combinations(range(len(tickers)), 2))

    candidates = pd.DataFrame({
        "ticker_a": [tickers[i] for i, _ in pairs],
        "ticker_b": [tickers[j] for _, j in pairs],
        "distance": distances,
        "overlap_days": len(formation),
    })

    candidates = candidates.replace(
        [np.inf, -np.inf], np.nan
    ).dropna(subset=["distance"])

    candidates = candidates.sort_values(
        "distance",
        ascending=True,
    ).reset_index(drop=True)

    candidates.insert(
        0,
        "rank",
        np.arange(1, len(candidates) + 1),
    )

    return candidates, formation


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prices = load_prices()
    candidates, formation = build_distance_candidates(prices)

    all_path = OUTPUT_DIR / "russell1000_baseline_all_pairs.csv"
    shortlist_path = (
        OUTPUT_DIR / "russell1000_baseline_candidates.csv"
    )

    candidates.to_csv(all_path, index=False)
    shortlist = candidates.head(TOP_N_PAIRS).copy()
    shortlist.to_csv(shortlist_path, index=False)

    print("\n--- RUSSELL 1000 BASELINE SUMMARY ---")
    print("Stocks used:", formation.shape[1])
    print("All unique pairs:", len(candidates))
    print("Shortlisted pairs:", len(shortlist))
    print("\nTop 20 candidate pairs:")
    print(shortlist.head(20).to_string(index=False))

    print("\nSaved:", all_path)
    print("Saved:", shortlist_path)


if __name__ == "__main__":
    main()
