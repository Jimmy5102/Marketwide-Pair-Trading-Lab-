
from pathlib import Path
from datetime import date

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

URL = (
    "https://raw.githubusercontent.com/"
    "Gary-Strauss/Russel_1000_Index_Constituents/"
    "master/data/russell1000_constituents.csv"
)


def main():
    print("Loading Russell 1000 constituent list...")

    source = pd.read_csv(URL)

    required = {
        "Company",
        "Symbol",
        "GICS_Sector",
    }

    missing = required - set(source.columns)
    if missing:
        raise ValueError(
            f"Missing expected source columns: {sorted(missing)}"
        )

    universe = pd.DataFrame({
        "ticker": source["Symbol"].astype(str).str.strip(),
        "company": source["Company"].astype(str).str.strip(),
        "sector": source["GICS_Sector"].fillna("Unknown").astype(str).str.strip(),
    })

    universe = universe[
        universe["ticker"].ne("")
        & universe["ticker"].ne("nan")
    ].copy()

    # Convert tickers to Yahoo Finance format.
    universe["yahoo_ticker"] = universe["ticker"].str.replace(
        ".", "-", regex=False
    )

    # Remove duplicate symbols and sort consistently.
    universe = (
        universe
        .drop_duplicates(subset="ticker")
        .sort_values("ticker")
        .reset_index(drop=True)
    )

    if not 900 <= len(universe) <= 1200:
        raise ValueError(
            f"Unexpected universe size: {len(universe)}. "
            "Inspect the source before downloading prices."
        )

    output = DATA_DIR / "universe_russell1000.csv"
    universe.to_csv(output, index=False)

    snapshot_date = date.today().isoformat()
    snapshot = DATA_DIR / (
        f"universe_russell1000_snapshot_{snapshot_date}.csv"
    )
    universe.to_csv(snapshot, index=False)

    print("\n--- UNIVERSE SUMMARY ---")
    print("Unique tickers:", universe["ticker"].nunique())
    print("Companies:", len(universe))
    print("\nSector counts:")
    print(universe["sector"].value_counts().to_string())
    print("\nFirst 10 rows:")
    print(universe.head(10).to_string(index=False))
    print("\nSaved:", output)
    print("Saved:", snapshot)


if __name__ == "__main__":
    main()
