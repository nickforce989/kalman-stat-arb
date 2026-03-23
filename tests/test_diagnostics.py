from __future__ import annotations

import pandas as pd

from quant_project.diagnostics import bootstrap_sharpe_ci, bootstrap_sharpe_difference


def test_bootstrap_sharpe_ci_contains_point_estimate() -> None:
    returns = pd.Series([0.01, 0.005, -0.002, 0.012, -0.003, 0.004, 0.008, -0.001])
    summary = bootstrap_sharpe_ci(returns, block_size=3, n_boot=80, seed=3)
    assert summary["ci_low"] <= summary["point_estimate"] <= summary["ci_high"]


def test_bootstrap_sharpe_difference_prefers_stronger_series() -> None:
    base = pd.Series([0.002, 0.003, -0.001, 0.004, 0.001, 0.002, 0.0, 0.003])
    stronger = base + 0.002
    summary = bootstrap_sharpe_difference(stronger, base, block_size=3, n_boot=80, seed=11)
    assert summary["probability_positive"] > 0.5
