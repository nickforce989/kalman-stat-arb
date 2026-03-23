from __future__ import annotations

import numpy as np
import pandas as pd

from quant_project.signals import (
    KalmanSignalConfig,
    build_kalman_signal,
    build_positions,
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


def test_kalman_signal_tracks_latent_spread_better_than_raw_observations() -> None:
    index = pd.RangeIndex(100)
    clean = pd.Series(np.sin(np.linspace(0.0, 4.0, 100)), index=index)
    noisy = clean + pd.Series(np.random.default_rng(7).normal(scale=0.35, size=100), index=index)

    kalman = build_kalman_signal(
        noisy,
        KalmanSignalConfig(process_var_multiplier=0.1, entry_z=1.5, exit_z=0.5, vol_span=10),
        observation_var=float(noisy.var()),
    ).frame

    raw_mse = float(((noisy - clean) ** 2).mean())
    kalman_mse = float(((kalman["anchor"] - clean) ** 2).mean())
    assert kalman_mse < raw_mse
