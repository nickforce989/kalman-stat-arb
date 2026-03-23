from __future__ import annotations

import pandas as pd

from quant_project.backtest import BacktestConfig, backtest_pair, summarize_returns


def test_backtest_charges_costs_on_position_changes() -> None:
    prices = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=4, freq="D"),
            "A": [100.0, 101.0, 102.0, 103.0],
            "B": [50.0, 50.0, 50.0, 50.0],
        }
    )
    positions = pd.Series([0.0, 1.0, 1.0, 0.0])
    result = backtest_pair(prices, "A", "B", beta=1.0, positions=positions, config=BacktestConfig(cost_bps=5.0))
    costs = result.frame["transaction_cost"].round(6).tolist()
    assert costs[0] == 0.0
    assert abs(costs[1] - 0.0005) < 1e-6
    assert costs[2] == 0.0
    assert abs(costs[3] - 0.000503) < 1e-6


def test_backtest_handles_time_varying_beta() -> None:
    prices = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "A": [100.0, 101.0, 103.0, 104.0, 105.0],
            "B": [50.0, 49.0, 48.0, 47.0, 46.0],
        }
    )
    positions = pd.Series([0.0, 1.0, 1.0, 1.0, 0.0])
    beta = pd.Series([1.0, 1.0, 0.8, 0.75, 0.75])

    result = backtest_pair(prices, "A", "B", beta=beta, positions=positions, config=BacktestConfig(cost_bps=5.0))
    assert result.frame["beta"].tolist() == beta.tolist()
    assert result.frame["notional_turnover"].iloc[2] > 0.0


def test_summarize_returns_reports_drawdown_and_trade_count() -> None:
    returns = pd.Series([0.01, -0.02, 0.0, 0.03])
    turnover = pd.Series([1.0, 1.0, 0.0, 2.0])
    notional_turnover = pd.Series([0.5, 0.1, 0.0, 0.8])
    metrics = summarize_returns(returns, turnover=turnover, notional_turnover=notional_turnover)
    assert round(metrics["max_drawdown"], 4) == -0.02
    assert metrics["trade_count"] == 2.0
    assert metrics["avg_notional_turnover"] == 0.35
