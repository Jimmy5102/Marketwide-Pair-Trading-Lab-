
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TEST_START = pd.Timestamp("2024-02-16")
STOP_Z = 4.0

def main():
    for model in ["baseline", "sector", "cluster"]:
        path = RESULTS / f"{model}_signals.csv"

        if not path.exists():
            print(f"\n{model}: no signal file")
            continue

        df = pd.read_csv(path, parse_dates=["date"])
        if df.empty:
            continue

        rows = []

        for (a, b), group in df.groupby(["ticker_a", "ticker_b"]):
            group = group.sort_values("date")
            z = group["zscore"].astype(float)

            first_date = group["date"].iloc[0]
            first_z = float(z.iloc[0])

            breaches = group.loc[z.abs() >= STOP_Z]

            rows.append({
                "ticker_a": a,
                "ticker_b": b,
                "first_test_date": first_date.date(),
                "first_test_z": first_z,
                "first_abs_z": abs(first_z),
                "first_stop_breach": (
                    breaches["date"].iloc[0].date()
                    if not breaches.empty else None
                ),
                "max_abs_z": float(z.abs().max()),
                "days_abs_z_ge_4": int((z.abs() >= STOP_Z).sum()),
                "days_abs_z_lt_2": int((z.abs() < 2).sum()),
            })

        report = pd.DataFrame(rows).sort_values(
            "first_abs_z", ascending=False
        )

        print(f"\n{'=' * 60}")
        print(f"{model.upper()} SPREAD STABILITY")
        print(report.to_string(index=False))

        output = RESULTS / f"{model}_spread_stability.csv"
        report.to_csv(output, index=False)
        print(f"Saved: {output}")

if __name__ == "__main__":
    main()