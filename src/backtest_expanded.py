from pathlib import Path

import numpy as np
import pandas as pd

PRICES_FILE = Path("data/processed/clean_russell1000_prices.csv")
SIGNALS_FILE = Path("results/russell1000_baseline_signals.csv")

RETURNS_FILE = Path(
    "results/russell1000_baseline_pair_backtest_returns.csv"
)
SUMMARY_FILE = Path(
    "results/russell1000_baseline_backtest_summary.csv"
)

TEST_START = "2024-02-16"
TEST_END = "2026-10-06"
COST_PER_LEG = 0.0005  # 5 bps per leg per unit of traded notional


def sharpe_ratio(returns):
    returns = returns.dropna()

    if len(returns) < 2 or returns.std(ddof=1) == 0:
        return np.nan

    return np.sqrt(252) * returns.mean() / returns.std(ddof=1)


def max_drawdown(equity):
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return drawdown.min()


def main():
    prices = pd.read_csv(
        PRICES_FILE, index_col=0, parse_dates=True
    ).sort_index()

    signals = pd.read_csv(
        SIGNALS_FILE, index_col=0, parse_dates=True
    ).sort_index()

    prices = prices.loc[~prices.index.duplicated(keep="last")]
    signals = signals.loc[~signals.index.duplicated(keep="last")]

    prices = prices.apply(pd.to_numeric, errors="coerce")
    daily_returns = prices.pct_change(fill_method=None)

    signals = signals.loc[
        (signals.index >= TEST_START)
        & (signals.index <= TEST_END)
    ]

    required = {
        "ticker_a",
        "ticker_b",
        "position",
        "formation_beta",
        "pair_retired",
    }

    missing = required - set(signals.columns)
    if missing:
        raise ValueError(
            f"Missing signal columns: {sorted(missing)}"
        )

    pair_returns = {}
    pair_turnovers = {}
    pair_summary = []

    for (ticker_a, ticker_b), group in signals.groupby(
        ["ticker_a", "ticker_b"], sort=True
    ):
        group = group.sort_index()

        if ticker_a not in daily_returns.columns:
            print(f"Skipping {ticker_a}-{ticker_b}: missing price column")
            continue

        if ticker_b not in daily_returns.columns:
            print(f"Skipping {ticker_a}-{ticker_b}: missing price column")
            continue

        if group.index.has_duplicates:
            raise ValueError(
                f"Duplicate signal dates for {ticker_a}-{ticker_b}"
            )

        beta_values = pd.to_numeric(
            group["formation_beta"], errors="coerce"
        ).dropna().unique()

        if len(beta_values) != 1:
            raise ValueError(
                f"Expected one fixed formation beta for "
                f"{ticker_a}-{ticker_b}; found {len(beta_values)}"
            )

        beta = float(beta_values[0])

        if not np.isfinite(beta):
            continue

        # Normalized dollar-neutral spread weights.
        weight_a = 1.0 / (1.0 + abs(beta))
        weight_b = -beta / (1.0 + abs(beta))

        # Signals use closing prices, so apply them from the next day.
        signal_position = pd.to_numeric(
            group["position"], errors="coerce"
        ).fillna(0)

        held_position = signal_position.shift(1).fillna(0)

        # Turnover occurs when the held position changes.
        turnover = held_position.diff().abs().fillna(0)

        ret_a = daily_returns[ticker_a].reindex(group.index)
        ret_b = daily_returns[ticker_b].reindex(group.index)

        gross_return = (
            held_position
            * (weight_a * ret_a + weight_b * ret_b)
        )

        # Each leg pays 5 bps on its traded notional.
        transaction_cost = (
            turnover
            * COST_PER_LEG
            * (abs(weight_a) + abs(weight_b))
        )

        net_return = gross_return - transaction_cost

        # Missing prices must not silently become zero returns.
        invalid = (
            ret_a.isna()
            | ret_b.isna()
            | gross_return.isna()
        )
        net_return = net_return.mask(invalid)

        pair_name = f"{ticker_a}-{ticker_b}"
        pair_returns[pair_name] = net_return
        pair_turnovers[pair_name] = turnover

        retired = (
            group["pair_retired"].astype(str).str.lower()
            .isin(["true", "1"])
        )

        pair_summary.append({
            "pair": pair_name,
            "ticker_a": ticker_a,
            "ticker_b": ticker_b,
            "formation_beta": beta,
            "test_days": len(group),
            "signal_position_changes": int(
                signal_position.diff().abs().fillna(0).sum()
            ),
            "retired": bool(retired.any()),
            "days_with_position": int((held_position != 0).sum()),
            "gross_cumulative_return": (
                (1 + gross_return.fillna(0)).prod() - 1
            ),
            "net_cumulative_return": (
                (1 + net_return.fillna(0)).prod() - 1
            ),
            "total_transaction_cost": transaction_cost.sum(),
        })

    if not pair_returns:
        raise RuntimeError("No valid pairs available for backtesting.")

    returns_df = pd.DataFrame(pair_returns).sort_index()

    # Treat unavailable pair returns as zero only for the portfolio
    # aggregation, while retaining the original pair-level return data.
    valid_pair_count = returns_df.notna().sum(axis=1)

    portfolio_returns = returns_df.fillna(0).mean(axis=1)

    # Keep only dates where at least one pair has a valid return.
    portfolio_returns = portfolio_returns.loc[valid_pair_count > 0]

    if portfolio_returns.empty:
        raise RuntimeError("No valid portfolio return observations.")

    equity = (1 + portfolio_returns).cumprod()

    returns_df["portfolio_net_return"] = portfolio_returns
    returns_df["portfolio_equity"] = equity

    RETURNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    returns_df.to_csv(RETURNS_FILE)

    summary = {
        "universe": "Russell 1000",
        "test_start": str(portfolio_returns.index.min().date()),
        "test_end": str(portfolio_returns.index.max().date()),
        "trading_days": len(portfolio_returns),
        "pairs_backtested": len(pair_returns),
        "cost_per_leg_bps": COST_PER_LEG * 10000,
        "portfolio_method": "fixed equal-weight pair sleeves",
        "cumulative_return": equity.iloc[-1] - 1,
        "annualized_return": (
            equity.iloc[-1] ** (252 / len(portfolio_returns)) - 1
            if equity.iloc[-1] > 0 else np.nan
        ),
        "annualized_volatility": (
            portfolio_returns.std(ddof=1) * np.sqrt(252)
        ),
        "sharpe_ratio": sharpe_ratio(portfolio_returns),
        "max_drawdown": max_drawdown(equity),
        "best_day": portfolio_returns.max(),
        "worst_day": portfolio_returns.min(),
        "positive_day_fraction": (
            portfolio_returns > 0
        ).mean(),
    }

    pair_summary_df = pd.DataFrame(pair_summary)
    pair_summary_df.to_csv(
        "results/russell1000_baseline_pair_summary.csv",
        index=False,
    )

    pd.DataFrame([summary]).to_csv(SUMMARY_FILE, index=False)

    print("\n--- PORTFOLIO BACKTEST SUMMARY ---")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\n--- PAIR RESULTS ---")
    print(
        pair_summary_df[
            [
                "pair",
                "signal_position_changes",
                "retired",
                "days_with_position",
                "net_cumulative_return",
                "total_transaction_cost",
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved: {RETURNS_FILE}")
    print(f"Saved: {SUMMARY_FILE}")
    print("Saved: results/russell1000_baseline_pair_summary.csv")


if __name__ == "__main__":
    main()
