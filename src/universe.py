
from pathlib import Path
from datetime import date

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

URL = (
    "https://raw.githubusercontent.com/"
    "datasets/s-and-p-500-companies/main/data/constituents.csv"
)


def main():
    universe = pd.read_csv(URL)

    universe.columns = [
        str(col).strip().lower().replace(" ", "_")
        for col in universe.columns
    ]

    ticker_col = next(
        (c for c in ["symbol", "ticker"] if c in universe.columns),
        None,
    )
    sector_col = next(
        (c for c in ["gics_sector", "sector"] if c in universe.columns),
        None,
    )
    name_col = next(
        (c for c in ["security", "name", "company"] if c in universe.columns),
        None,
    )

    if ticker_col is None or sector_col is None:
        raise ValueError(
            f"Unexpected source columns: {universe.columns.tolist()}"
        )

    result = pd.DataFrame({
        "ticker": universe[ticker_col].astype(str).str.strip(),
        "sector": universe[sector_col].astype(str).str.strip(),
    })

    if name_col is not None:
        result["company"] = universe[name_col].astype(str).str.strip()

    result = result.drop_duplicates("ticker")
    result = result[result["ticker"].ne("")].copy()

    # Yahoo Finance uses BRK-B rather than BRK.B.
    result["yahoo_ticker"] = result["ticker"].str.replace(
        ".", "-", regex=False
    )

    if result["ticker"].nunique() < 450:
        raise ValueError(
            "Too few constituents were downloaded. "
            "Check the data source before proceeding."
        )

    result = result.sort_values("ticker").reset_index(drop=True)

    result.to_csv(DATA_DIR / "universe.csv", index=False)

    snapshot_date = date.today().isoformat()
    result.to_csv(
        DATA_DIR / f"universe_snapshot_{snapshot_date}.csv",
        index=False,
    )

    print("Current constituent securities:", len(result))
    print("\nSector counts:")
    print(result["sector"].value_counts().to_string())
    print("\nSaved data/universe.csv")
    print(f"Saved dated snapshot for {snapshot_date}")


if __name__ == "__main__":
    main()
