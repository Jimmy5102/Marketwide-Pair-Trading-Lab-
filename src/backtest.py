
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRICE_PATH = ROOT / "data" / "processed" / "clean_prices.csv"
RESULTS_DIR = ROOT / "results"

FORMATION_END = "2023-12-29"
TEST_START = "2024-02-16"
TEST_END = "2026-10-06"

# Assumed one-way trading cost, per leg, in basis points.
COST_BPS_PER_LEG = 5.0

MODEL_FILES = {
    "baseline": "baseline_signals.csv",
    "sector": "sector_signals.csv",
    "cluster": "cluster_signals.csv",
}


def load_prices():
    prices = pd.read_csv(
        PRICE_PATH, index_col=0, parse_dates=True
    ).sort_index()

    prices = prices.loc[~prices.index.duplicated(keep="last")]
    return prices.apply(pd.to_numeric, errors="coerce")


def backtest_pair(prices, signals):
    ticker_a = signals["ticker_a"].iloc[0]
    ticker_b = signals["ticker_b"].iloc[0]
    beta = float(signals["formation_beta"].iloc[0])

    pair_prices = prices[[ticker_a, ticker_b]].dropna()

    # Returns are measured over the interval from previous close
    # to current close.
    asset_returns = pair_prices.pct_change(fill_method=None)

    # Match the signal dates to the common price calendar.
    signals = signals.copy()
    signals.index = pd.to_datetime(signals.index)
    signals = signals[~signals.index.duplicated(keep="last")]

    positions = signals["position"].reindex(pair_prices.index).fillna(0)

    # A signal observed at the close of day t is traded from t+1.
    held_position = positions.shift(1).fillna(0)

    # Dollar-neutral weights scaled so the gross exposure sums to 1.
    gross_a = 1.0
    gross_b = abs(beta)
    total_gross = gross_a + gross_b

    weight_a = 1.0 / total_gross
    weight_b = -beta / total_gross

    # Long spread = long A and short beta-adjusted B.
    pair_gross_return = held_position * (
        weight_a * asset_returns[ticker_a]
        + weight_b * asset_returns[ticker_b]
    )

    # Turnover is measured from changes in the spread position.
    # Each entry/exit changes exposure in both stock legs.
    position_change = held_position.diff().abs()
    position_change.iloc[0] = abs(held_position.iloc[0])

    turnover = position_change * (abs(weight_a) + abs(weight_b))
    cost_rate = COST_BPS_PER_LEG / 10000.0

    # Charge the one-way cost on the turnover of each leg.
    transaction_cost = turnover * cost_rate

    pair_net_return = pair_gross_return - transaction_cost

    result = pd.DataFrame({
        "gross_return": pair_gross_return,
        "transaction_cost": transaction_cost,
        "net_return": pair_net_return,
        "held_position": held_position,
        "turnover": turnover,
    }, index=pair_prices.index)

    result = result.loc[
        (result.index >= pd.Timestamp(TEST_START))
        & (result.index <= pd.Timestamp(TEST_END))
    ].copy()

    result["ticker_a"] = ticker_a
    result["ticker_b"] = ticker_b

    return result


def calculate_metrics(returns):
    returns = returns.dropna()

    if len(returns) < 2:
        return {
            "trading_days": len(returns),
            "total_return": np.nan,
            "cagr": np.nan,
            "annualized_volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
        }

    equity = (1 + returns).cumprod()
    total_return = equity.iloc[-1] - 1

    years = len(returns) / 252
    cagr = (
        equity.iloc[-1] ** (1 / years) - 1
        if years > 0 and equity.iloc[-1] > 0
        else np.nan
    )

    volatility = returns.std(ddof=1) * np.sqrt(252)
    mean_return = returns.mean() * 252

    sharpe = (
        mean_return / volatility
        if volatility > 0
        else np.nan
    )

    drawdown = equity / equity.cummax() - 1

    return {
        "trading_days": len(returns),
        "total_return": total_return,
        "cagr": cagr,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": drawdown.min(),
    }


def main():
    if not PRICE_PATH.exists():
        raise FileNotFoundError(f"Missing {PRICE_PATH}")

    prices = load_prices()
    all_pair_results = []
    model_summaries = []

    for model_name, signal_filename in MODEL_FILES.items():
        signal_path = RESULTS_DIR / signal_filename

        if not signal_path.exists():
            print(f"\n{model_name}: signal file missing; skipping.")
            continue

        signals = pd.read_csv(
            signal_path, parse_dates=["date"]
        ).set_index("date")

        if signals.empty:
            print(f"\n{model_name}: no signals; skipping.")
            continue

        pair_results = []

        grouped = signals.groupby(["ticker_a", "ticker_b"])

        for (ticker_a, ticker_b), pair_signals in grouped:
            try:
                result = backtest_pair(prices, pair_signals)
                result["model"] = model_name
                pair_results.append(result)
            except (KeyError, ValueError, IndexError) as exc:
                print(f"Skipped {ticker_a}/{ticker_b}: {exc}")

        if not pair_results:
            print(f"\n{model_name}: no valid pair backtests.")
            model_summaries.append({
                "model": model_name,
                "pairs_tested": 0,
                "total_return": np.nan,
                "cagr": np.nan,
                "annualized_volatility": np.nan,
                "sharpe": np.nan,
                "max_drawdown": np.nan,
                "total_trades": 0,
            })
            continue

        model_results = pd.concat(pair_results)
        all_pair_results.append(model_results)

        # Equal-weight the pair returns. This is a simple research
        # portfolio; overlapping stocks can create duplicated exposure.
        pair_returns = model_results.pivot_table(
            index=model_results.index,
            columns=["ticker_a", "ticker_b"],
            values="net_return",
            aggfunc="first",
        )

        portfolio_returns = pair_returns.mean(axis=1, skipna=True)
        portfolio_returns = portfolio_returns.fillna(0)

        metrics = calculate_metrics(portfolio_returns)

        # Count entries: transitions from flat to non-flat exposure.
        trade_count = int(
            (
                model_results["held_position"]
                .groupby([model_results["ticker_a"],
                          model_results["ticker_b"]])
                .diff()
                .fillna(model_results["held_position"])
                .ne(0)
            ).sum()
        )

        model_summaries.append({
            "model": model_name,
            "pairs_tested": len(pair_results),
            **metrics,
            "total_trades": trade_count,
            "mean_daily_transaction_cost": (
                model_results["transaction_cost"].mean()
            ),
        })

        print(f"\n--- {model_name.upper()} ---")
        for key, value in metrics.items():
            print(f"{key}: {value}")
        print("Pairs tested:", len(pair_results))
        print("Trade transitions:", trade_count)

    if all_pair_results:
        pd.concat(all_pair_results).to_csv(
            RESULTS_DIR / "pair_backtest_returns.csv",
            index_label="date",
        )

    summary = pd.DataFrame(model_summaries)
    summary.to_csv(
        RESULTS_DIR / "backtest_model_comparison.csv",
        index=False,
    )

    print("\n--- MODEL COMPARISON ---")
    print(summary.to_string(index=False))
    print("\nSaved results/backtest_model_comparison.csv")


if __name__ == "__main__":
    main()
