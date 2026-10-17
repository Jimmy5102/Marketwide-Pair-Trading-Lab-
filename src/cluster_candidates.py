
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PRICE_PATH = ROOT / "data" / "processed" / "clean_prices.csv"
OUTPUT_DIR = ROOT / "results"

FORMATION_END = "2023-12-29"
FORMATION_DAYS = 252

N_CLUSTERS = 20
TOP_N_PAIRS = 500
RANDOM_STATE = 42


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not PRICE_PATH.exists():
        raise FileNotFoundError(
            f"Missing {PRICE_PATH}. Complete Step 2 first."
        )

    prices = pd.read_csv(
        PRICE_PATH, index_col=0, parse_dates=True
    ).sort_index()

    formation = prices.loc[:FORMATION_END].tail(FORMATION_DAYS)

    if len(formation) < FORMATION_DAYS:
        raise ValueError(
            f"Only {len(formation)} formation dates available."
        )

    # Use the same complete stock set and dates for all models.
    formation = formation.loc[:, formation.notna().all(axis=0)]
    formation = formation.loc[:, (formation > 0).all(axis=0)]

    if formation.shape[1] < N_CLUSTERS:
        raise ValueError(
            f"Only {formation.shape[1]} stocks available; "
            f"cannot create {N_CLUSTERS} clusters."
        )

    # Daily returns avoid clustering raw price levels.
    returns = formation.pct_change(fill_method=None).iloc[1:]

    # Standardize each stock's return series so clustering focuses
    # on pattern rather than each stock's absolute volatility.
    scaler = StandardScaler()
    features = scaler.fit_transform(returns.T)

    model = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=20,
    )

    labels = model.fit_predict(features)

    assignments = pd.DataFrame({
        "ticker": returns.columns,
        "cluster": labels,
    })

    assignments.to_csv(
        OUTPUT_DIR / "cluster_assignments.csv",
        index=False,
    )

    # Use normalized prices to rank pairs within each cluster.
    normalized = formation.div(formation.iloc[0], axis=1)
    pairs = []

    cluster_map = assignments.set_index("ticker")["cluster"]

    for cluster_id, group in assignments.groupby("cluster"):
        tickers = sorted(group["ticker"].tolist())

        if len(tickers) < 2:
            continue

        for ticker_a, ticker_b in combinations(tickers, 2):
            path_a = normalized[ticker_a].to_numpy()
            path_b = normalized[ticker_b].to_numpy()

            distance = float(np.sum((path_a - path_b) ** 2))

            pairs.append({
                "ticker_a": ticker_a,
                "ticker_b": ticker_b,
                "cluster": int(cluster_id),
                "distance": distance,
                "overlap_days": len(formation),
                "model": "cluster",
            })

    if not pairs:
        raise RuntimeError(
            "No within-cluster pairs generated. Try fewer clusters."
        )

    candidates = pd.DataFrame(pairs).sort_values(
        "distance", ascending=True
    ).reset_index(drop=True)

    candidates.insert(0, "rank", np.arange(1, len(candidates) + 1))

    all_path = OUTPUT_DIR / "cluster_all_pairs.csv"
    shortlist_path = OUTPUT_DIR / "cluster_candidates.csv"

    candidates.to_csv(all_path, index=False)
    candidates.head(TOP_N_PAIRS).to_csv(
        shortlist_path, index=False
    )

    print("\n--- CLUSTER MODEL SUMMARY ---")
    print("Stocks clustered:", len(assignments))
    print("Number of clusters requested:", N_CLUSTERS)
    print("Cluster sizes:")
    print(assignments["cluster"].value_counts().sort_index().to_string())
    print("Within-cluster candidate pairs:", len(candidates))
    print("Shortlisted pairs:", min(TOP_N_PAIRS, len(candidates)))
    print("\nTop 20 pairs:")
    print(candidates.head(20).to_string(index=False))
    print("\nSaved:", OUTPUT_DIR / "cluster_assignments.csv")
    print("Saved:", all_path)
    print("Saved:", shortlist_path)


if __name__ == "__main__":
    main()
