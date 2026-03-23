from __future__ import annotations

import numpy as np
import pandas as pd

from quant_project.signals import (
    KalmanRegressionSignalConfig,
    build_kalman_regression_signal,
    build_positions,
    estimate_dynamic_hedge_ratio,
    estimate_hedge_ratio,
)


def test_build_positions_enters_and_exits_symmetrically() -> None:
    zscore = pd.Series([0.0, 1.6, 1.2, 0.4, -1.7, -1.2, -0.2])
    positions = build_positions(zscore, entry_z=1.5, exit_z=0.5)
    assert positions.tolist() == [0.0, -1.0, -1.0, 0.0, 1.0, 1.0, 0.0]


def test_estimate_hedge_ratio_recovers_linear_relationship() -> None:
    asset_b = pd.Series(np.linspace(10.0, 20.0, 100))
    asset_a = 3.0 + 0.75 * asset_b
    hedge = estimate_hedge_ratio(asset_a, asset_b)
    assert abs(hedge.alpha - 3.0) < 1e-10
    assert abs(hedge.beta - 0.75) < 1e-10


def test_dynamic_hedge_ratio_tracks_drifting_beta_better_than_static_beta() -> None:
    rng = np.random.default_rng(7)
    asset_b = pd.Series(np.linspace(90.0, 120.0, 160) + rng.normal(scale=1.0, size=160))
    true_beta = pd.Series(np.linspace(0.7, 1.25, 160))
    asset_a = 4.0 + true_beta * asset_b + rng.normal(scale=0.75, size=160)

    dynamic = estimate_dynamic_hedge_ratio(asset_a, asset_b, train_size=100, process_var_multiplier=0.01)
    static_beta = estimate_hedge_ratio(asset_a.iloc[:100], asset_b.iloc[:100]).beta

    dynamic_error = float(np.mean(np.abs(dynamic["beta"] - true_beta)))
    static_error = float(np.mean(np.abs(static_beta - true_beta)))
    assert dynamic_error < static_error


def test_kalman_regression_signal_exposes_dynamic_beta_column() -> None:
    asset_b = pd.Series(np.linspace(50.0, 60.0, 120))
    asset_a = pd.Series(5.0 + 1.1 * asset_b + np.sin(np.linspace(0.0, 3.0, 120)))

    signal = build_kalman_regression_signal(
        asset_a,
        asset_b,
        train_size=80,
        config=KalmanRegressionSignalConfig(process_var_multiplier=0.01, entry_z=1.5, exit_z=0.5, vol_span=20),
    ).frame

    assert {"alpha", "beta", "spread", "signal_vol", "zscore", "position"}.issubset(signal.columns)
    assert signal["beta"].notna().all()
