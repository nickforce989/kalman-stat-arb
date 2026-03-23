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
class KalmanRegressionSignalConfig:
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
    residual = (spread - anchor).rename("spread")
    scale = spread.rolling(config.window).std().replace(0.0, np.nan).rename("signal_vol")
    zscore = (residual / scale).rename("zscore")
    positions = build_positions(zscore, config.entry_z, config.exit_z)

    frame = pd.DataFrame(
        {
            "spread": residual,
            "anchor": pd.Series(0.0, index=spread.index, name="anchor"),
            "signal_vol": scale,
            "zscore": zscore,
            "position": positions,
        }
    )
    return SignalFrame(frame=frame)


def _safe_scale(values: pd.Series) -> float:
    scale = float(values.std(ddof=0))
    return scale if scale > 1e-8 else 1.0


def estimate_dynamic_hedge_ratio(
    asset_a: pd.Series,
    asset_b: pd.Series,
    train_size: int,
    process_var_multiplier: float,
) -> pd.DataFrame:
    """Estimate time-varying alpha_t and beta_t using a Kalman regression."""
    if train_size <= 1 or train_size > len(asset_a):
        raise ValueError("train_size must leave at least one observation for sequential estimation.")

    asset_a = asset_a.reset_index(drop=True).astype(float)
    asset_b = asset_b.reset_index(drop=True).astype(float)

    train_a = asset_a.iloc[:train_size]
    train_b = asset_b.iloc[:train_size]
    mean_a = float(train_a.mean())
    mean_b = float(train_b.mean())
    scale_a = _safe_scale(train_a)
    scale_b = _safe_scale(train_b)

    asset_a_norm = (asset_a - mean_a) / scale_a
    asset_b_norm = (asset_b - mean_b) / scale_b

    design_train = np.column_stack([np.ones(train_size), train_b.to_numpy()])
    initial_theta_price, _, _, _ = np.linalg.lstsq(design_train, train_a.to_numpy(), rcond=None)
    alpha_price_init = float(initial_theta_price[0])
    beta_price_init = float(initial_theta_price[1])
    beta_norm_init = beta_price_init * scale_b / scale_a
    alpha_norm_init = (alpha_price_init + beta_price_init * mean_b - mean_a) / scale_a
    theta = np.array([alpha_norm_init, beta_norm_init], dtype=float)

    static_fit_norm = alpha_norm_init + beta_norm_init * asset_b_norm.iloc[:train_size].to_numpy()
    observation_residual = asset_a_norm.iloc[:train_size].to_numpy() - static_fit_norm
    observation_var = max(float(np.var(observation_residual)), 1e-6)
    process_var = max(process_var_multiplier * observation_var, 1e-9)

    covariance = np.eye(2, dtype=float)
    process_covariance = process_var * np.eye(2, dtype=float)
    alpha_path_norm = np.zeros(len(asset_a), dtype=float)
    beta_path_norm = np.zeros(len(asset_a), dtype=float)
    fitted_norm = np.zeros(len(asset_a), dtype=float)

    for idx in range(len(asset_a)):
        theta_prior = theta
        covariance_prior = covariance + process_covariance
        observation_vector = np.array([1.0, asset_b_norm.iloc[idx]], dtype=float)
        prediction = float(observation_vector @ theta_prior)
        innovation = float(asset_a_norm.iloc[idx] - prediction)
        innovation_var = float(observation_vector @ covariance_prior @ observation_vector.T + observation_var)
        kalman_gain = covariance_prior @ observation_vector / innovation_var

        theta = theta_prior + kalman_gain * innovation
        covariance = covariance_prior - np.outer(kalman_gain, observation_vector) @ covariance_prior

        alpha_path_norm[idx] = theta[0]
        beta_path_norm[idx] = theta[1]
        fitted_norm[idx] = float(observation_vector @ theta)

    beta_path = scale_a * beta_path_norm / scale_b
    alpha_path = mean_a + scale_a * alpha_path_norm - beta_path * mean_b
    fitted_price = alpha_path + beta_path * asset_b.to_numpy()
    spread = asset_a.to_numpy() - fitted_price

    return pd.DataFrame(
        {
            "alpha": alpha_path,
            "beta": beta_path,
            "fitted_price": fitted_price,
            "spread": spread,
        },
        index=asset_a.index,
    )


def build_kalman_regression_signal(
    asset_a: pd.Series,
    asset_b: pd.Series,
    train_size: int,
    config: KalmanRegressionSignalConfig,
) -> SignalFrame:
    regression_frame = estimate_dynamic_hedge_ratio(
        asset_a=asset_a,
        asset_b=asset_b,
        train_size=train_size,
        process_var_multiplier=config.process_var_multiplier,
    )
    spread = regression_frame["spread"].rename("spread")
    scale = spread.ewm(span=config.vol_span, adjust=False).std().replace(0.0, np.nan).rename("signal_vol")
    zscore = (spread / scale).rename("zscore")
    positions = build_positions(zscore, config.entry_z, config.exit_z)
    anchor = pd.Series(0.0, index=spread.index, name="anchor")

    frame = regression_frame.copy()
    frame["anchor"] = anchor
    frame["signal_vol"] = scale
    frame["zscore"] = zscore
    frame["position"] = positions
    return SignalFrame(frame=frame)
