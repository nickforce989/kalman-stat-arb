from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    annualization: int = 252
    cost_bps: float = 5.0


@dataclass
class BacktestResult:
    frame: pd.DataFrame
    metrics: dict[str, float]


def summarize_returns(
    returns: pd.Series,
    turnover: pd.Series | None = None,
    annualization: int = 252,
) -> dict[str, float]:
    clean_returns = returns.fillna(0.0)
    equity = (1.0 + clean_returns).cumprod()
    total_return = float(equity.iloc[-1] - 1.0) if len(equity) else 0.0
    ann_return = float((1.0 + total_return) ** (annualization / len(clean_returns)) - 1.0) if len(clean_returns) else 0.0
    ann_vol = float(clean_returns.std(ddof=0) * np.sqrt(annualization))
    sharpe = float(clean_returns.mean() / clean_returns.std(ddof=0) * np.sqrt(annualization)) if ann_vol > 0 else 0.0
    drawdown = equity / equity.cummax() - 1.0 if len(equity) else pd.Series(dtype=float)
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0
    trade_count = float(turnover.fillna(0.0).sum() / 2.0) if turnover is not None else 0.0
    active_days = float((clean_returns != 0.0).mean()) if len(clean_returns) else 0.0

    return {
        "annual_return": ann_return,
        "annual_volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "total_return": total_return,
        "trade_count": trade_count,
        "active_day_ratio": active_days,
    }


def backtest_pair(
    prices: pd.DataFrame,
    asset_a: str,
    asset_b: str,
    beta: float,
    positions: pd.Series,
    config: BacktestConfig,
) -> BacktestResult:
    """Backtest a one-unit spread where asset_a is hedged by beta units of asset_b."""
    asset_a_prices = prices[asset_a].reset_index(drop=True)
    asset_b_prices = prices[asset_b].reset_index(drop=True)
    aligned_positions = positions.reset_index(drop=True).fillna(0.0)

    spread_pnl = asset_a_prices.diff() - beta * asset_b_prices.diff()
    gross_notional = asset_a_prices.shift(1).abs() + abs(beta) * asset_b_prices.shift(1).abs()
    gross_notional = gross_notional.replace(0.0, np.nan)

    gross_return = aligned_positions.shift(1).fillna(0.0) * spread_pnl / gross_notional
    turnover = aligned_positions.diff().abs().fillna(aligned_positions.abs())
    transaction_cost = turnover * (2.0 * config.cost_bps / 10_000.0)
    net_return = (gross_return - transaction_cost).fillna(0.0)
    equity = (1.0 + net_return).cumprod()

    frame = prices.copy()
    frame["gross_return"] = gross_return.fillna(0.0)
    frame["transaction_cost"] = transaction_cost.fillna(0.0)
    frame["net_return"] = net_return
    frame["equity"] = equity
    frame["turnover"] = turnover.fillna(0.0)

    metrics = summarize_returns(frame["net_return"], turnover=frame["turnover"], annualization=config.annualization)
    return BacktestResult(frame=frame, metrics=metrics)
