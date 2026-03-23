from __future__ import annotations

from dataclasses import dataclass

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
    notional_turnover: pd.Series | None = None,
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
    avg_notional_turnover = float(notional_turnover.fillna(0.0).mean()) if notional_turnover is not None else 0.0
    active_days = float((clean_returns != 0.0).mean()) if len(clean_returns) else 0.0

    return {
        "annual_return": ann_return,
        "annual_volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "total_return": total_return,
        "trade_count": trade_count,
        "avg_notional_turnover": avg_notional_turnover,
        "active_day_ratio": active_days,
    }


def _coerce_beta_series(beta: float | pd.Series, length: int) -> pd.Series:
    if isinstance(beta, pd.Series):
        if len(beta) != length:
            raise ValueError("Dynamic beta series must have the same length as the price frame.")
        return beta.reset_index(drop=True).astype(float)
    return pd.Series(float(beta), index=pd.RangeIndex(length), dtype=float)


def backtest_pair(
    prices: pd.DataFrame,
    asset_a: str,
    asset_b: str,
    beta: float | pd.Series,
    positions: pd.Series,
    config: BacktestConfig,
) -> BacktestResult:
    """Backtest a one-unit spread, allowing for a time-varying hedge ratio."""
    asset_a_prices = prices[asset_a].reset_index(drop=True)
    asset_b_prices = prices[asset_b].reset_index(drop=True)
    aligned_positions = positions.reset_index(drop=True).fillna(0.0)
    beta_series = _coerce_beta_series(beta, len(prices)).ffill().fillna(0.0)

    asset_a_return_exposure = aligned_positions
    asset_b_return_exposure = -(aligned_positions * beta_series)
    prev_asset_a_exposure = asset_a_return_exposure.shift(1).fillna(0.0)
    prev_asset_b_exposure = asset_b_return_exposure.shift(1).fillna(0.0)

    asset_a_delta = asset_a_prices.diff().fillna(0.0)
    asset_b_delta = asset_b_prices.diff().fillna(0.0)
    spread_pnl = prev_asset_a_exposure * asset_a_delta + prev_asset_b_exposure * asset_b_delta

    previous_gross_notional = (
        prev_asset_a_exposure.abs() * asset_a_prices.shift(1) + prev_asset_b_exposure.abs() * asset_b_prices.shift(1)
    )
    current_gross_notional = asset_a_return_exposure.abs() * asset_a_prices + asset_b_return_exposure.abs() * asset_b_prices
    pnl_denominator = previous_gross_notional.replace(0.0, np.nan)
    gross_return = spread_pnl / pnl_denominator

    traded_notional = (
        (asset_a_return_exposure - prev_asset_a_exposure).abs() * asset_a_prices
        + (asset_b_return_exposure - prev_asset_b_exposure).abs() * asset_b_prices
    )
    cost_denominator = previous_gross_notional.where(previous_gross_notional > 0.0, current_gross_notional).replace(
        0.0, np.nan
    )
    notional_turnover = traded_notional / cost_denominator
    transaction_cost = notional_turnover.fillna(0.0) * (config.cost_bps / 10_000.0)

    position_turnover = aligned_positions.diff().abs().fillna(aligned_positions.abs())
    net_return = (gross_return.fillna(0.0) - transaction_cost).fillna(0.0)
    equity = (1.0 + net_return).cumprod()

    frame = prices.copy()
    frame["beta"] = beta_series
    frame["gross_return"] = gross_return.fillna(0.0)
    frame["transaction_cost"] = transaction_cost.fillna(0.0)
    frame["net_return"] = net_return
    frame["equity"] = equity
    frame["turnover"] = position_turnover.fillna(0.0)
    frame["notional_turnover"] = notional_turnover.fillna(0.0)

    metrics = summarize_returns(
        frame["net_return"],
        turnover=frame["turnover"],
        notional_turnover=frame["notional_turnover"],
        annualization=config.annualization,
    )
    return BacktestResult(frame=frame, metrics=metrics)
