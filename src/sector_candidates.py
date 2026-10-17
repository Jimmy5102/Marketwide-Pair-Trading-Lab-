
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRICE_PATH = ROOT / "data" / "processed" / "clean_prices.csv"
UNIVERSE_PATH = ROOT / "data" / "processed" / "eligible_universe.csv"
OUTPUT_DIR = ROOT / "results"

FORMATION_END = "2023-12-29"
FORMATION_DAYS = 252
TOP_N_PAIRS = 500


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for path in [PRICE_PATH, UNIVERSE_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Complete Steps 2 and 3 first."
            )

    prices = pd.read_csv(
        PRICE_PATH, index_col=0, parse_dates=True
    ).sort_index()

    universe = pd.read_csv(UNIVERSE_PATH)
    formation = prices.loc[:FORMATION_END].tail(FORMATION_DAYS)

    if len(formation) < FORMATION_DAYS:
        raise ValueError(
            f"Only {len(formation)} formation dates available."
        )

    # Require the same common, complete formation window.
    complete_tickers = formation.columns[
        formation.notna().all(axis=0)
        & (formation > 0).all(axis=0)
    ].tolist()

    universe = universe[
        universe["ticker"].isin(complete_tickers)
    ].drop_duplicates("ticker")

    # Normalize paths so distance is comparable across share prices.
    normalized = formation[complete_tickers].div(
        formation[complete_tickers].iloc[0], axis=1
    )

    pairs = []

    for sector, group in universe.groupby("sector"):
        tickers = sorted(group["ticker"].unique())

        for ticker_a, ticker_b in combinations(tickers, 2):
            path_a = normalized[ticker_a].to_numpy()
            path_b = normalized[ticker_b].to_numpy()

            distance = float(np.sum((path_a - path_b) ** 2))

            pairs.append({
                "ticker_a": ticker_a,
                "ticker_b": ticker_b,
                "sector": sector,
                "distance": distance,
                "overlap_days": len(formation),
                "model": "sector",
            })

    if not pairs:
        raise RuntimeError(
            "No same-sector pairs generated. Check sector labels."
        )

    candidates = pd.DataFrame(pairs)
    candidates = candidates.sort_values(
        "distance", ascending=True
    ).reset_index(drop=True)

    candidates.insert(0, "rank", np.arange(1, len(candidates) + 1))

    all_path = OUTPUT_DIR / "sector_all_pairs.csv"
    shortlist_path = OUTPUT_DIR / "sector_candidates.csv"

    candidates.to_csv(all_path, index=False)
    candidates.head(TOP_N_PAIRS).to_csv(
        shortlist_path, index=False
    )

    print("\n--- SECTOR MODEL SUMMARY ---")
    print("Eligible stocks in formation window:", len(complete_tickers))
    print("Same-sector candidate pairs:", len(candidates))
    print("Shortlisted pairs:", min(TOP_N_PAIRS, len(candidates)))
    print("\nPairs by sector:")
    print(candidates["sector"].value_counts().to_string())
    print("\nTop 20 pairs:")
    print(candidates.head(20).to_string(index=False))
    print("\nSaved:", all_path)
    print("Saved:", shortlist_path)


if __name__ == "__main__":
    main()
