from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HedgeRatio:
    alpha: float
    beta: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class NaiveSignalConfig:
    window: int
    entry_z: float
    exit_z: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class KalmanSignalConfig:
    process_var_multiplier: float
    entry_z: float
    exit_z: float
    vol_span: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass
class SignalFrame:
    frame: pd.DataFrame


def estimate_hedge_ratio(asset_a: pd.Series, asset_b: pd.Series) -> HedgeRatio:
    """Estimate a static hedge ratio on the training sample."""
    beta, alpha = np.polyfit(asset_b.to_numpy(), asset_a.to_numpy(), 1)
    return HedgeRatio(alpha=float(alpha), beta=float(beta))


def compute_spread(asset_a: pd.Series, asset_b: pd.Series, hedge_ratio: HedgeRatio) -> pd.Series:
    return asset_a - (hedge_ratio.alpha + hedge_ratio.beta * asset_b)


def kalman_local_level(
    observations: pd.Series,
    process_var: float,
    observation_var: float,
) -> tuple[pd.Series, pd.Series]:
    """Run a one-dimensional local-level Kalman filter."""
    values = observations.to_numpy(dtype=float)
    filtered = np.zeros(len(values), dtype=float)
    posterior_var = np.zeros(len(values), dtype=float)

    filtered[0] = values[0]
    posterior_var[0] = observation_var

    for idx in range(1, len(values)):
        prior_mean = filtered[idx - 1]
        prior_var = posterior_var[idx - 1] + process_var
        gain = prior_var / (prior_var + observation_var)
        filtered[idx] = prior_mean + gain * (values[idx] - prior_mean)
        posterior_var[idx] = (1.0 - gain) * prior_var

    return (
        pd.Series(filtered, index=observations.index, name="anchor"),
        pd.Series(posterior_var, index=observations.index, name="anchor_var"),
    )


def build_positions(zscore: pd.Series, entry_z: float, exit_z: float) -> pd.Series:
    """Trade the spread mean reversion using simple finite-state logic."""
    state = 0
    positions: list[int] = []

    for value in zscore.fillna(0.0):
        if state == 0:
            if value > entry_z:
                state = -1
            elif value < -entry_z:
                state = 1
        elif state == 1 and value >= -exit_z:
            state = 0
        elif state == -1 and value <= exit_z:
            state = 0
        positions.append(state)

    return pd.Series(positions, index=zscore.index, name="position", dtype=float)


def build_naive_signal(spread: pd.Series, config: NaiveSignalConfig) -> SignalFrame:
    anchor = spread.rolling(config.window).mean().rename("anchor")
    residual = (spread - anchor).rename("residual")
    scale = spread.rolling(config.window).std().replace(0.0, np.nan).rename("signal_vol")
    zscore = (residual / scale).rename("zscore")
    positions = build_positions(zscore, config.entry_z, config.exit_z)

    frame = pd.DataFrame(
        {
            "spread": spread,
            "anchor": anchor,
            "residual": residual,
            "signal_vol": scale,
            "zscore": zscore,
            "position": positions,
        }
    )
    return SignalFrame(frame=frame)


def build_kalman_signal(
    spread: pd.Series,
    config: KalmanSignalConfig,
    observation_var: float,
) -> SignalFrame:
    process_var = max(config.process_var_multiplier * observation_var, 1e-12)
    anchor, anchor_var = kalman_local_level(spread, process_var=process_var, observation_var=observation_var)
    residual = (spread - anchor).rename("residual")
    scale = residual.ewm(span=config.vol_span, adjust=False).std().replace(0.0, np.nan).rename("signal_vol")
    zscore = (residual / scale).rename("zscore")
    positions = build_positions(zscore, config.entry_z, config.exit_z)

    frame = pd.DataFrame(
        {
            "spread": spread,
            "anchor": anchor,
            "anchor_var": anchor_var,
            "residual": residual,
            "signal_vol": scale,
            "zscore": zscore,
            "position": positions,
        }
    )
    return SignalFrame(frame=frame)
