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
    costs = result.frame["transaction_cost"].tolist()
    assert costs == [0.0, 0.001, 0.0, 0.001]


def test_summarize_returns_reports_drawdown_and_trade_count() -> None:
    returns = pd.Series([0.01, -0.02, 0.0, 0.03])
    turnover = pd.Series([1.0, 1.0, 0.0, 2.0])
    metrics = summarize_returns(returns, turnover=turnover)
    assert round(metrics["max_drawdown"], 4) == -0.02
    assert metrics["trade_count"] == 2.0
