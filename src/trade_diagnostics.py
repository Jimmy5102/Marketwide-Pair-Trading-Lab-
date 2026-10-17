
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

ENTRY_Z = 2.0
STOP_Z = 4.0

def main():
    for model in ["baseline", "sector", "cluster"]:
        path = RESULTS / f"{model}_signals.csv"

        if not path.exists():
            print(f"\n{model}: no signal file")
            continue

        signals = pd.read_csv(path, parse_dates=["date"])
        if signals.empty:
            print(f"\n{model}: empty signal file")
            continue

        print(f"\n{'=' * 60}")
        print(f"MODEL: {model.upper()}")

        summaries = []

        for (ticker_a, ticker_b), group in signals.groupby(
            ["ticker_a", "ticker_b"]
        ):
            group = group.sort_values("date").copy()

            position = group["position"].astype(int)
            previous = position.shift(1).fillna(0).astype(int)

            # A transition includes entries, exits, and reversals.
            transitions = int(position.ne(previous).sum())

            # Count dates at or beyond the stop threshold.
            stop_days = int(
                group["zscore"].abs().ge(STOP_Z).sum()
            )

            # A position is entered while the z-score is beyond entry.
            entry_days = int(
                (
                    (previous == 0)
                    & (position != 0)
                ).sum()
            )

            summaries.append({
                "ticker_a": ticker_a,
                "ticker_b": ticker_b,
                "days": len(group),
                "transitions": transitions,
                "entries": entry_days,
                "stop_threshold_days": stop_days,
                "mean_abs_z": group["zscore"].abs().mean(),
                "max_abs_z": group["zscore"].abs().max(),
                "long_days": int((position == 1).sum()),
                "short_days": int((position == -1).sum()),
            })

        summary = pd.DataFrame(summaries).sort_values(
            "transitions", ascending=False
        )

        print(summary.to_string(index=False))
        print("\nTotal transitions:", summary["transitions"].sum())
        print("Total entries:", summary["entries"].sum())
        print("Total stop-threshold days:",
              summary["stop_threshold_days"].sum())

if __name__ == "__main__":
    main()
