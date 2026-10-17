
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

TEST_START = "2024-02-16"
TEST_END = "2026-10-06"

ENTRY_Z = 2.0
EXIT_Z = 0.0
STOP_Z = 4.0


# --------------------------------------------------
# Generate signals for one pair
# --------------------------------------------------

def make_pair_signals(
    prices,
    ticker_a,
    ticker_b,
    intercept,
    beta,
):
    """
    Generate out-of-sample trading signals for one pair.

    Position convention:
        +1 = long spread
        -1 = short spread
         0 = flat

    Entry:
        z-score <= -ENTRY_Z: long spread
        z-score >= +ENTRY_Z: short spread

    Exit:
        Long spread exits when z-score >= EXIT_Z.
        Short spread exits when z-score <= EXIT_Z.

    Stop:
        If an active position reaches abs(z-score) >= STOP_Z,
        close the position and retire the pair for the rest
        of the test period.

    Formation-period spread mean and standard deviation are
    fixed throughout the test period.
    """

    # ----------------------------------------------
    # Validate input
    # ----------------------------------------------

    if ticker_a not in prices.columns:
        raise ValueError(f"Missing ticker: {ticker_a}")

    if ticker_b not in prices.columns:
        raise ValueError(f"Missing ticker: {ticker_b}")

    if not np.isfinite(intercept) or not np.isfinite(beta):
        raise ValueError("Intercept and beta must be finite.")

    if beta <= 0:
        raise ValueError("Expected a positive hedge ratio.")

    pair = prices[[ticker_a, ticker_b]].copy()
    pair = pair.sort_index()
    pair = pair.loc[~pair.index.duplicated(keep="last")]
    pair = pair.dropna()

    if (pair <= 0).any().any():
        raise ValueError("Prices must be strictly positive.")

    # ----------------------------------------------
    # Formation period
    # ----------------------------------------------

    formation = pair.loc[:FORMATION_END].tail(FORMATION_DAYS)

    if len(formation) != FORMATION_DAYS:
        raise ValueError(
            f"{ticker_a}-{ticker_b}: expected "
            f"{FORMATION_DAYS} formation observations, "
            f"found {len(formation)}."
        )

    formation_spread = (
        np.log(formation[ticker_a])
        - intercept
        - beta * np.log(formation[ticker_b])
    )

    spread_mean = float(formation_spread.mean())
    spread_std = float(formation_spread.std(ddof=1))

    if not np.isfinite(spread_mean):
        raise ValueError("Formation spread mean is not finite.")

    if not np.isfinite(spread_std) or spread_std <= 0:
        raise ValueError("Formation spread standard deviation is invalid.")

    # ----------------------------------------------
    # Calculate spread and z-score
    # ----------------------------------------------

    spread = (
        np.log(pair[ticker_a])
        - intercept
        - beta * np.log(pair[ticker_b])
    )

    zscore = (spread - spread_mean) / spread_std

    # Restrict signals to the specified test period.
    test_mask = (
        (zscore.index >= pd.Timestamp(TEST_START))
        & (zscore.index <= pd.Timestamp(TEST_END))
    )

    test_zscore = zscore.loc[test_mask]
    test_spread = spread.reindex(test_zscore.index)

    if test_zscore.empty:
        raise ValueError(
            f"{ticker_a}-{ticker_b}: no observations in the test period."
        )

    # ----------------------------------------------
    # Position state machine
    # ----------------------------------------------

    positions = np.zeros(len(test_zscore), dtype=int)

    current_position = 0
    pair_retired = False

    for i, z in enumerate(test_zscore.to_numpy()):

        if pair_retired:
            current_position = 0

        elif not np.isfinite(z):
            current_position = 0

        elif current_position == 0:
            # Do not enter a spread already beyond the stop boundary.
            if abs(z) >= STOP_Z:
                pair_retired = True

            elif z >= ENTRY_Z:
                current_position = -1

            elif z <= -ENTRY_Z:
                current_position = 1

        elif abs(z) >= STOP_Z:
            # Close the open position and retire this pair.
            current_position = 0
            pair_retired = True

        elif current_position == 1 and z >= EXIT_Z:
            current_position = 0

        elif current_position == -1 and z <= EXIT_Z:
            current_position = 0

        positions[i] = current_position

    # ----------------------------------------------
    # Build output
    # ----------------------------------------------

    result = pd.DataFrame(
        {
            "ticker_a": ticker_a,
            "ticker_b": ticker_b,
            "spread": test_spread,
            "zscore": test_zscore,
            "position": positions,
        },
        index=test_zscore.index,
    )

    result["stop_threshold"] = STOP_Z
    result["pair_retired"] = pair_retired

    return result


# --------------------------------------------------
# Generate signals for all candidate models
# --------------------------------------------------

def main():

    if not PRICE_PATH.exists():
        raise FileNotFoundError(
            f"Price file not found: {PRICE_PATH}. "
            "Run the data preparation steps first."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    prices = pd.read_csv(
        PRICE_PATH,
        index_col=0,
        parse_dates=True,
    ).sort_index()

    prices = prices.loc[
        ~prices.index.duplicated(keep="last")
    ]

    prices = prices.apply(pd.to_numeric, errors="coerce")

    model_files = {
        "baseline": "baseline_screened_pairs.csv",
        "sector": "sector_screened_pairs.csv",
        "cluster": "cluster_screened_pairs.csv",
    }

    for model_name, filename in model_files.items():

        pair_path = RESULTS_DIR / filename

        if not pair_path.exists():
            print(
                f"{model_name}: screened-pair file missing; "
                "skipping."
            )
            continue

        pairs = pd.read_csv(pair_path)

        if pairs.empty:
            print(
                f"{model_name}: no eligible screened pairs; "
                "no signal file created."
            )
            continue

        required_columns = {
            "ticker_a",
            "ticker_b",
            "intercept",
            "hedge_ratio",
        }

        missing_columns = required_columns - set(pairs.columns)

        if missing_columns:
            raise ValueError(
                f"{filename} is missing columns: "
                f"{sorted(missing_columns)}"
            )

        outputs = []
        retired_pairs = 0

        print(
            f"\nGenerating {model_name} signals "
            f"for {len(pairs)} pairs..."
        )

        for _, row in pairs.iterrows():

            try:
                signals = make_pair_signals(
                    prices=prices,
                    ticker_a=row["ticker_a"],
                    ticker_b=row["ticker_b"],
                    intercept=float(row["intercept"]),
                    beta=float(row["hedge_ratio"]),
                )

                signals["model"] = model_name
                signals["formation_intercept"] = float(
                    row["intercept"]
                )
                signals["formation_beta"] = float(
                    row["hedge_ratio"]
                )

                if bool(signals["pair_retired"].iloc[0]):
                    retired_pairs += 1

                outputs.append(signals)

            except (ValueError, KeyError) as exc:
                print(
                    f"  Skipping {row['ticker_a']}-"
                    f"{row['ticker_b']}: {exc}"
                )

        if not outputs:
            print(
                f"{model_name}: no signals generated "
                "successfully."
            )
            continue

        combined = pd.concat(outputs)
        combined.index.name = "date"

        output_path = RESULTS_DIR / f"{model_name}_signals.csv"
        combined.to_csv(output_path)

        print(f"{model_name}: generated signals for {len(outputs)} pairs")
        print(f"Saved: {output_path}")
        print(f"Pairs retired after a stop: {retired_pairs}")
        print(
            "Date range:",
            combined.index.min().date(),
            "to",
            combined.index.max().date(),
        )


if __name__ == "__main__":
    main()
