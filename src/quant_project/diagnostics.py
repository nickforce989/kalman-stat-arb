from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint, kpss


def _coerce_series(values: pd.Series) -> pd.Series:
    return values.reset_index(drop=True).astype(float).dropna()


def _series_sharpe(returns: np.ndarray, annualization: int) -> float:
    volatility = returns.std(ddof=0)
    if volatility <= 0.0:
        return 0.0
    return float(np.sqrt(annualization) * returns.mean() / volatility)


def _moving_block_bootstrap_indices(length: int, block_size: int, n_boot: int, seed: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("Bootstrap requires at least one observation.")

    rng = np.random.default_rng(seed)
    block_size = max(1, min(block_size, length))
    block_count = int(np.ceil(length / block_size))
    max_start = max(length - block_size + 1, 1)

    draws = np.empty((n_boot, length), dtype=int)
    for boot_idx in range(n_boot):
        starts = rng.integers(0, max_start, size=block_count)
        indices = np.concatenate([np.arange(start, start + block_size) for start in starts])[:length]
        draws[boot_idx] = indices
    return draws


def bootstrap_sharpe_ci(
    returns: pd.Series,
    annualization: int = 252,
    block_size: int = 20,
    n_boot: int = 300,
    seed: int = 7,
) -> dict[str, float | int]:
    clean_returns = _coerce_series(returns).to_numpy()
    point_estimate = _series_sharpe(clean_returns, annualization)

    if len(clean_returns) < 2:
        return {
            "point_estimate": point_estimate,
            "ci_low": point_estimate,
            "ci_high": point_estimate,
            "probability_positive": float(point_estimate > 0.0),
            "block_size": block_size,
            "n_boot": n_boot,
        }

    bootstrap_indices = _moving_block_bootstrap_indices(len(clean_returns), block_size, n_boot, seed)
    samples = np.array([_series_sharpe(clean_returns[index_set], annualization) for index_set in bootstrap_indices])

    return {
        "point_estimate": point_estimate,
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "probability_positive": float((samples > 0.0).mean()),
        "block_size": block_size,
        "n_boot": n_boot,
    }


def bootstrap_sharpe_difference(
    left_returns: pd.Series,
    right_returns: pd.Series,
    annualization: int = 252,
    block_size: int = 20,
    n_boot: int = 300,
    seed: int = 7,
) -> dict[str, float | int]:
    left = _coerce_series(left_returns)
    right = _coerce_series(right_returns)
    length = min(len(left), len(right))
    if length == 0:
        return {
            "point_estimate": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "probability_positive": 0.0,
            "block_size": block_size,
            "n_boot": n_boot,
        }

    left_values = left.iloc[-length:].to_numpy()
    right_values = right.iloc[-length:].to_numpy()
    point_estimate = _series_sharpe(left_values, annualization) - _series_sharpe(right_values, annualization)
    bootstrap_indices = _moving_block_bootstrap_indices(length, block_size, n_boot, seed)
    samples = np.array(
        [
            _series_sharpe(left_values[index_set], annualization) - _series_sharpe(right_values[index_set], annualization)
            for index_set in bootstrap_indices
        ]
    )
    return {
        "point_estimate": float(point_estimate),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "probability_positive": float((samples > 0.0).mean()),
        "block_size": block_size,
        "n_boot": n_boot,
    }


def _format_stationarity_output(result: tuple[Any, ...]) -> dict[str, Any]:
    statistic, pvalue, usedlag, nobs, critical_values = result[:5]
    return {
        "statistic": float(statistic),
        "pvalue": float(pvalue),
        "used_lag": int(usedlag),
        "observations": int(nobs),
        "critical_values": {key: float(value) for key, value in critical_values.items()},
    }


def _run_adf(series: pd.Series) -> dict[str, Any]:
    return _format_stationarity_output(adfuller(_coerce_series(series), regression="c", autolag="AIC"))


def _run_kpss(series: pd.Series) -> dict[str, Any]:
    statistic, pvalue, usedlag, critical_values = kpss(_coerce_series(series), regression="c", nlags="auto")
    return {
        "statistic": float(statistic),
        "pvalue": float(pvalue),
        "used_lag": int(usedlag),
        "critical_values": {key: float(value) for key, value in critical_values.items()},
    }


def run_pair_diagnostics(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    asset_a: str,
    asset_b: str,
) -> dict[str, Any]:
    train_target = train_frame[asset_a].reset_index(drop=True)
    train_regressor = train_frame[asset_b].reset_index(drop=True)
    test_target = test_frame[asset_a].reset_index(drop=True)
    test_regressor = test_frame[asset_b].reset_index(drop=True)

    design = sm.add_constant(train_regressor.rename(asset_b))
    ols_fit = sm.OLS(train_target, design).fit()
    alpha = float(ols_fit.params["const"])
    beta = float(ols_fit.params[asset_b])
    conf_int = ols_fit.conf_int(alpha=0.05)

    train_spread = train_target - (alpha + beta * train_regressor)
    test_spread = test_target - (alpha + beta * test_regressor)
    coint_stat, coint_pvalue, critical_values = coint(train_target, train_regressor)

    return {
        "train_ols": {
            "alpha": alpha,
            "beta": beta,
            "beta_ci_95": [
                float(conf_int.loc[asset_b, 0]),
                float(conf_int.loc[asset_b, 1]),
            ],
            "r_squared": float(ols_fit.rsquared),
        },
        "engle_granger": {
            "statistic": float(coint_stat),
            "pvalue": float(coint_pvalue),
            "critical_values": [float(value) for value in critical_values],
        },
        "spread_stationarity": {
            "train_adf": _run_adf(train_spread),
            "train_kpss": _run_kpss(train_spread),
            "test_adf": _run_adf(test_spread),
            "test_kpss": _run_kpss(test_spread),
        },
    }
