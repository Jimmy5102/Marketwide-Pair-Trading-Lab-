
from pathlib import Path
import time

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

START_DATE = "2021-01-01"
END_DATE = "2026-10-17"  # Exclusive: includes data through Oct 16
MIN_OBSERVATIONS = 252
BATCH_SIZE = 50

for folder in [RAW_DIR, PROCESSED_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

def download_prices(tickers):
    frames = []
    failed = []

    # Remove null, non-string, and empty ticker values.
    tickers = [
        str(t).strip()
        for t in tickers
        if pd.notna(t) and str(t).strip()
        and str(t).strip().lower() != "nan"
    ]
    tickers = list(dict.fromkeys(tickers))

    def download_one(ticker):
        """Download one ticker and return a one-column DataFrame."""
        data = yf.download(
            tickers=ticker,
            start=START_DATE,
            end=END_DATE,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if data.empty or "Close" not in data.columns:
            return None

        close = data["Close"]

        # Handle either a Series or a one-column DataFrame.
        if isinstance(close, pd.DataFrame):
            if close.empty:
                return None
            close = close.iloc[:, 0]

        result = close.to_frame(name=ticker)
        result.index = pd.to_datetime(result.index)
        return result

    for start in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[start:start + BATCH_SIZE]
        batch_number = start // BATCH_SIZE + 1

        print(
            f"Downloading batch {batch_number}: "
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
                raise RuntimeError("Batch returned no data")

            if isinstance(data.columns, pd.MultiIndex):
                if "Close" not in data.columns.get_level_values(0):
                    raise RuntimeError("Close prices missing in batch")
                close = data["Close"].copy()
            else:
                # A single-ticker response may have flat columns.
                if "Close" not in data.columns:
                    raise RuntimeError("Close prices missing in batch")
                close = data[["Close"]].copy()
                close.columns = batch[:1]

            frames.append(close)

        except Exception as exc:
            print(f"Batch {batch_number} failed: {repr(exc)}")
            print("Retrying tickers individually...")

            for ticker in batch:
                try:
                    single = download_one(ticker)

                    if single is None or single.empty:
                        print(f"  No usable data: {ticker}")
                        failed.append(ticker)
                    else:
                        frames.append(single)

                except Exception as single_exc:
                    print(f"  Failed ticker {ticker}: {repr(single_exc)}")
                    failed.append(ticker)

                time.sleep(0.2)

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

    failed = sorted(set(str(t).strip() for t in failed))
    return prices, failed

def main():
    universe_path = DATA_DIR / "universe_russell1000.csv"

    if not universe_path.exists():
        raise FileNotFoundError(
            f"Universe file not found: {universe_path}. "
            "Run python -m src.universe_expanded first."
        )

    universe = pd.read_csv(universe_path)
    universe = universe.drop_duplicates("ticker").copy()

    universe["ticker"] = universe["ticker"].astype(str).str.strip()
    universe["yahoo_ticker"] = (
        universe["yahoo_ticker"].astype(str).str.strip()
    )

    universe = universe[
        universe["ticker"].ne("")
        & universe["yahoo_ticker"].ne("")
        & universe["ticker"].ne("nan")
        & universe["yahoo_ticker"].ne("nan")
    ].copy()

    ticker_map = dict(
        zip(universe["yahoo_ticker"], universe["ticker"])
    )

    tickers = list(universe["yahoo_ticker"]) + ["SPY"]
    tickers = list(dict.fromkeys(tickers))

    print("Russell 1000 universe rows:", len(universe))
    print("Tickers to download, including SPY:", len(tickers))
    print(f"Date range: {START_DATE} to {END_DATE} (exclusive)")

    
    prices, failed = download_prices(tickers)
    prices = prices.rename(columns=ticker_map)

    # If SPY is absent from the batch results, retry it separately.
    if "SPY" not in prices.columns or prices["SPY"].notna().sum() == 0:
        print("SPY missing from batch results; downloading separately.")

        try:
            spy = yf.download(
                tickers="SPY",
                start=START_DATE,
                end=END_DATE,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )

            if spy.empty or "Close" not in spy.columns:
                raise RuntimeError("SPY returned no usable Close prices")

            spy_close = spy["Close"]
            if isinstance(spy_close, pd.DataFrame):
                spy_close = spy_close.iloc[:, 0]

            prices["SPY"] = spy_close.reindex(prices.index)

        except Exception as exc:
            raise RuntimeError(
                f"SPY reference download failed: {exc}"
            ) from exc

    if prices["SPY"].notna().sum() == 0:
        raise RuntimeError(
            "SPY reference data is still missing. "
            "Check Yahoo Finance connectivity before continuing."
        )


    prices.to_csv(RAW_DIR / "russell1000_downloaded_close_prices.csv")

    # Use the reference ETF's valid trading dates as the calendar.
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

    quality["valid_observations"] = (
        quality["ticker"].map(observations).fillna(0).astype(int)
    )
    quality["coverage"] = quality["ticker"].map(coverage).fillna(0.0)

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
        PROCESSED_DIR / "russell1000_data_quality_report.csv",
        index=False,
    )

    # Save the full matrix so excluded stocks remain auditable.
    stock_prices.to_csv(
        PROCESSED_DIR / "all_russell1000_prices.csv"
    )

    eligible = quality.loc[quality["eligible"], "ticker"].tolist()
    clean_prices = stock_prices[eligible]

    clean_prices.to_csv(
        PROCESSED_DIR / "clean_russell1000_prices.csv"
    )

    clean_prices.pct_change(fill_method=None).to_csv(
        PROCESSED_DIR / "russell1000_daily_returns.csv"
    )

    universe[universe["ticker"].isin(eligible)][
        ["ticker", "sector"]
    ].to_csv(
        PROCESSED_DIR / "eligible_russell1000_universe.csv",
        index=False,
    )

    pd.Series(failed, name="failed_yahoo_ticker").to_csv(
        PROCESSED_DIR / "russell1000_failed_batches.csv",
        index=False,
    )

    print("\n--- RUSSELL 1000 DATA SUMMARY ---")
    print("Constituent rows:", len(universe))
    print("Reference trading days:", len(calendar))
    print("Full price matrix:", stock_prices.shape)
    print("Eligible stocks:", len(eligible))
    print("Failed batch tickers:", len(failed))
    print("Date range:", calendar.min(), "to", calendar.max())
    print(
        "Missing prices in full matrix:",
        int(stock_prices.isna().sum().sum()),
    )
    print(
        "Median stock coverage:",
        round(float(quality["coverage"].median()), 3),
    )

    if len(eligible) < 700:
        print(
            "\nWARNING: Fewer than 700 stocks passed the initial "
            "filter. Inspect the quality report before continuing."
        )

    print("\nSaved Russell 1000 files:")
    print("- data/raw/russell1000_downloaded_close_prices.csv")
    print("- data/processed/all_russell1000_prices.csv")
    print("- data/processed/clean_russell1000_prices.csv")
    print("- data/processed/russell1000_daily_returns.csv")
    print("- data/processed/eligible_russell1000_universe.csv")
    print("- data/processed/russell1000_data_quality_report.csv")


if __name__ == "__main__":
    main()
