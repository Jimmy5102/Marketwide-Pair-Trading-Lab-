
from pathlib import Path
import time

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

START_DATE = "2021-01-01"
END_DATE = "2026-10-16"  # Exclusive
MIN_OBSERVATIONS = 252
BATCH_SIZE = 50

for folder in [RAW_DIR, PROCESSED_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


def download_prices(tickers):
    frames = []
    failed = []

    for start in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[start:start + BATCH_SIZE]
        print(
            f"Downloading batch {start // BATCH_SIZE + 1}: "
            f"{len(batch)} tickers"
        )

        try:
            data = yf.download(
                tickers=batch,
                start=START_DATE,
                end=END_DATE,
                interval="1d",
                auto_adjust=True,
                group_by="column",
                threads=True,
                progress=False,
            )

            if data.empty:
                failed.extend(batch)
                continue

            if isinstance(data.columns, pd.MultiIndex):
                close = data["Close"].copy()
            else:
                close = data[["Close"]].copy()
                close.columns = batch[:1]

            frames.append(close)

        except Exception as exc:
            print("Batch failed:", exc)
            failed.extend(batch)

        time.sleep(1)

    if not frames:
        raise RuntimeError(
            "No prices were downloaded. Check internet access "
            "and Yahoo Finance availability."
        )

    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]

    return prices, failed


def main():
    universe = pd.read_csv(DATA_DIR / "universe.csv")
    universe = universe.drop_duplicates("ticker")

    ticker_map = dict(
        zip(universe["yahoo_ticker"], universe["ticker"])
    )

    tickers = list(universe["yahoo_ticker"]) + ["SPY"]
    tickers = list(dict.fromkeys(tickers))

    prices, failed = download_prices(tickers)
    prices = prices.rename(columns=ticker_map)

    if "SPY" not in prices.columns:
        raise RuntimeError(
            "SPY reference data is missing. Retry the download."
        )

    prices.to_csv(RAW_DIR / "downloaded_close_prices.csv")

    # Use dates where the reference ETF has a valid observation.
    calendar = prices.index[prices["SPY"].notna()]
    stock_tickers = universe["ticker"].tolist()

    stock_prices = prices.reindex(calendar).reindex(
        columns=stock_tickers
    )

    observations = stock_prices.notna().sum()
    coverage = stock_prices.notna().mean()

    quality = universe[
        ["ticker", "yahoo_ticker", "sector"]
    ].copy()

    quality["valid_observations"] = quality["ticker"].map(observations)
    quality["coverage"] = quality["ticker"].map(coverage)
    quality["first_valid_date"] = [
        stock_prices[t].first_valid_index()
        for t in quality["ticker"]
    ]
    quality["last_valid_date"] = [
        stock_prices[t].last_valid_index()
        for t in quality["ticker"]
    ]
    quality["download_failed"] = quality["yahoo_ticker"].isin(failed)
    quality["eligible"] = (
        (quality["valid_observations"] >= MIN_OBSERVATIONS)
        & ~quality["download_failed"]
    )

    quality.to_csv(
        PROCESSED_DIR / "data_quality_report.csv",
        index=False,
    )

    # Keep the full matrix so we can audit exclusions.
    stock_prices.to_csv(
        PROCESSED_DIR / "all_sp500_prices.csv"
    )

    eligible = quality.loc[quality["eligible"], "ticker"].tolist()
    clean_prices = stock_prices[eligible]

    clean_prices.to_csv(
        PROCESSED_DIR / "clean_prices.csv"
    )

    clean_prices.pct_change(fill_method=None).to_csv(
        PROCESSED_DIR / "daily_returns.csv"
    )

    universe[universe["ticker"].isin(eligible)][
        ["ticker", "sector"]
    ].to_csv(
        PROCESSED_DIR / "eligible_universe.csv",
        index=False,
    )

    print("\n--- DATA SUMMARY ---")
    print("Constituent securities:", len(universe))
    print("Reference trading days:", len(calendar))
    print("Full price matrix:", stock_prices.shape)
    print("Eligible stocks:", len(eligible))
    print("Download failures:", len(failed))
    print("Date range:", calendar.min(), "to", calendar.max())
    print("Missing prices in full matrix:", stock_prices.isna().sum().sum())

    if failed:
        print("\nReview failed tickers in the quality report.")
    if len(eligible) < 400:
        print(
            "\nWARNING: Fewer than 400 stocks passed the initial "
            "filter. Inspect the report before continuing."
        )


if __name__ == "__main__":
    main()
