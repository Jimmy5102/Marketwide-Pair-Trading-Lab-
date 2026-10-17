from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRICE_PATH = ROOT / "data" / "processed" / "clean_russell1000_prices.csv"
PAIRS_PATH = ROOT / "results" / "russell1000_baseline_stable_pairs.csv"
OUTPUT_PATH = ROOT / "results" / "russell1000_baseline_signals.csv"

FORMATION_END = "2023-12-29"
FORMATION_DAYS = 252
TEST_START = "2024-02-16"
TEST_END = "2026-10-06"

ENTRY_Z = 2.0
EXIT_Z = 0.0
STOP_Z = 4.0


def main():
    prices = pd.read_csv(
        PRICE_PATH, index_col=0, parse_dates=True
    ).sort_index()

    prices = prices.loc[~prices.index.duplicated(keep="last")]
    prices = prices.apply(pd.to_numeric, errors="coerce")
    pairs = pd.read_csv(PAIRS_PATH)

    outputs = []

    print("Stable pairs:", len(pairs))

    for _, row in pairs.iterrows():
        a = row["ticker_a"]
        b = row["ticker_b"]
        beta = float(row["hedge_ratio"])
        intercept = float(row["intercept"])

        pair = prices[[a, b]].dropna()

        if (pair <= 0).any().any():
            print(f"Skipping {a}-{b}: non-positive price")
            continue

        formation = pair.loc[:FORMATION_END].tail(FORMATION_DAYS)

        if len(formation) != FORMATION_DAYS:
            print(f"Skipping {a}-{b}: incomplete formation data")
            continue

        formation_spread = (
            np.log(formation[a]) - intercept - beta * np.log(formation[b])
        )
        spread_mean = float(formation_spread.mean())
        spread_std = float(formation_spread.std(ddof=1))

        if not np.isfinite(spread_std) or spread_std <= 0:
            print(f"Skipping {a}-{b}: invalid spread standard deviation")
            continue

        spread = np.log(pair[a]) - intercept - beta * np.log(pair[b])
        zscore = (spread - spread_mean) / spread_std

        mask = (
            (zscore.index >= pd.Timestamp(TEST_START))
            & (zscore.index <= pd.Timestamp(TEST_END))
        )
        zscore = zscore.loc[mask]
        spread = spread.loc[mask]

        if zscore.empty:
            print(f"Skipping {a}-{b}: no test observations")
            continue

        position = 0
        retired = False
        positions = []
        retired_flags = []

        for z in zscore.to_numpy():
            if retired or not np.isfinite(z):
                position = 0

            elif position == 0:
                if abs(z) >= STOP_Z:
                    retired = True
                elif z >= ENTRY_Z:
                    position = -1
                elif z <= -ENTRY_Z:
                    position = 1

            elif abs(z) >= STOP_Z:
                position = 0
                retired = True

            elif position == 1 and z >= EXIT_Z:
                position = 0

            elif position == -1 and z <= EXIT_Z:
                position = 0

            positions.append(position)
            retired_flags.append(retired)

        frame = pd.DataFrame({
            "ticker_a": a,
            "ticker_b": b,
            "spread": spread,
            "zscore": zscore,
            "position": positions,
            "pair_retired": retired_flags,
            "formation_intercept": intercept,
            "formation_beta": beta,
        }, index=zscore.index)

        outputs.append(frame)

        entries = int(
            (pd.Series(positions, index=zscore.index).diff().abs() == 1).sum()
        )
        print(
            f"{a}-{b}: {len(frame)} test days, "
            f"{entries} position transitions, retired={retired}"
        )

    if not outputs:
        raise RuntimeError("No signal series were generated.")

    result = pd.concat(outputs).sort_index()
    result.index.name = "date"
    result.to_csv(OUTPUT_PATH)

    print("\n--- SIGNAL SUMMARY ---")
    print("Pairs with signals:", len(outputs))
    print("Signal rows:", len(result))
    print("Pairs retired:", result.groupby(
        ["ticker_a", "ticker_b"]
    )["pair_retired"].max().sum())
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
