from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from quant_project.backtest import BacktestConfig, backtest_pair, summarize_returns
from quant_project.data import load_pair_close_frame
from quant_project.signals import (
    HedgeRatio,
    KalmanSignalConfig,
    NaiveSignalConfig,
    SignalFrame,
    build_kalman_signal,
    build_naive_signal,
    compute_spread,
    estimate_hedge_ratio,
)


@dataclass(frozen=True)
class ResearchConfig:
    asset_a: str = "V"
    asset_b: str = "MA"
    start_date: str = "2012-01-01"
    train_ratio: float = 0.6
    cost_bps: float = 5.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StrategyResearchResult:
    name: str
    config: dict[str, float | int]
    full_frame: pd.DataFrame
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    tuning_grid: pd.DataFrame


@dataclass
class ResearchResult:
    config: ResearchConfig
    hedge_ratio: HedgeRatio
    pair_frame: pd.DataFrame
    split_index: int
    naive: StrategyResearchResult
    kalman: StrategyResearchResult

    def to_summary_dict(self) -> dict[str, Any]:
        split_date = self.pair_frame.loc[self.split_index, "Date"].strftime("%Y-%m-%d")
        return {
            "research_config": self.config.to_dict(),
            "hedge_ratio": self.hedge_ratio.to_dict(),
            "split_index": self.split_index,
            "train_end_date": self.pair_frame.loc[self.split_index - 1, "Date"].strftime("%Y-%m-%d"),
            "test_start_date": split_date,
            "naive": {
                "config": self.naive.config,
                "train_metrics": self.naive.train_metrics,
                "test_metrics": self.naive.test_metrics,
            },
            "kalman": {
                "config": self.kalman.config,
                "train_metrics": self.kalman.train_metrics,
                "test_metrics": self.kalman.test_metrics,
            },
        }


def _select_better_candidate(
    best: tuple[tuple[float, float, float], Any] | None,
    candidate_rank: tuple[float, float, float],
    payload: Any,
) -> tuple[tuple[float, float, float], Any]:
    if best is None or candidate_rank > best[0]:
        return candidate_rank, payload
    return best


def _evaluate_signal(
    pair_frame: pd.DataFrame,
    asset_a: str,
    asset_b: str,
    beta: float,
    signal: SignalFrame,
    cost_bps: float,
) -> pd.DataFrame:
    backtest = backtest_pair(
        prices=pair_frame.loc[:, ["Date", asset_a, asset_b]],
        asset_a=asset_a,
        asset_b=asset_b,
        beta=beta,
        positions=signal.frame["position"],
        config=BacktestConfig(cost_bps=cost_bps),
    )
    enriched = pair_frame.copy()
    for column in signal.frame.columns:
        enriched[column] = signal.frame[column].to_numpy()
    for column in ["gross_return", "transaction_cost", "net_return", "equity", "turnover"]:
        enriched[column] = backtest.frame[column].to_numpy()
    return enriched


def _slice_metrics(frame: pd.DataFrame) -> dict[str, float]:
    return summarize_returns(frame["net_return"], turnover=frame["turnover"])


def _tune_naive_strategy(
    train_frame: pd.DataFrame,
    pair_frame: pd.DataFrame,
    asset_a: str,
    asset_b: str,
    hedge_ratio: HedgeRatio,
    cost_bps: float,
) -> StrategyResearchResult:
    spread_train = compute_spread(train_frame[asset_a], train_frame[asset_b], hedge_ratio)
    best: tuple[tuple[float, float, float], NaiveSignalConfig] | None = None
    rows: list[dict[str, float | int]] = []

    for window in [10, 20, 40, 60]:
        for entry_z in [1.0, 1.25, 1.5, 1.75, 2.0]:
            for exit_z in [0.0, 0.25, 0.5, 0.75]:
                if exit_z >= entry_z:
                    continue
                config = NaiveSignalConfig(window=window, entry_z=entry_z, exit_z=exit_z)
                signal_train = build_naive_signal(spread_train, config)
                evaluation = _evaluate_signal(train_frame, asset_a, asset_b, hedge_ratio.beta, signal_train, cost_bps)
                metrics = _slice_metrics(evaluation)
                rows.append({**config.to_dict(), **metrics})
                rank = (metrics["sharpe"], metrics["total_return"], -metrics["max_drawdown"])
                best = _select_better_candidate(best, rank, config)

    if best is None:
        raise RuntimeError("Failed to tune naive strategy.")

    best_config = best[1]
    spread_full = compute_spread(pair_frame[asset_a], pair_frame[asset_b], hedge_ratio)
    signal_full = build_naive_signal(spread_full, best_config)
    full_frame = _evaluate_signal(pair_frame, asset_a, asset_b, hedge_ratio.beta, signal_full, cost_bps)
    split_index = len(train_frame)

    return StrategyResearchResult(
        name="rolling_zscore",
        config=best_config.to_dict(),
        full_frame=full_frame,
        train_metrics=_slice_metrics(full_frame.iloc[:split_index].reset_index(drop=True)),
        test_metrics=_slice_metrics(full_frame.iloc[split_index:].reset_index(drop=True)),
        tuning_grid=pd.DataFrame(rows),
    )


def _tune_kalman_strategy(
    train_frame: pd.DataFrame,
    pair_frame: pd.DataFrame,
    asset_a: str,
    asset_b: str,
    hedge_ratio: HedgeRatio,
    cost_bps: float,
) -> StrategyResearchResult:
    spread_train = compute_spread(train_frame[asset_a], train_frame[asset_b], hedge_ratio)
    observation_var = max(float(spread_train.var()), 1e-12)
    best: tuple[tuple[float, float, float], KalmanSignalConfig] | None = None
    rows: list[dict[str, float | int]] = []

    for process_var_multiplier in [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]:
        for entry_z in [1.0, 1.25, 1.5, 1.75, 2.0]:
            for exit_z in [0.0, 0.25, 0.5, 0.75]:
                if exit_z >= entry_z:
                    continue
                for vol_span in [10, 20, 40, 60]:
                    config = KalmanSignalConfig(
                        process_var_multiplier=process_var_multiplier,
                        entry_z=entry_z,
                        exit_z=exit_z,
                        vol_span=vol_span,
                    )
                    signal_train = build_kalman_signal(spread_train, config, observation_var=observation_var)
                    evaluation = _evaluate_signal(train_frame, asset_a, asset_b, hedge_ratio.beta, signal_train, cost_bps)
                    metrics = _slice_metrics(evaluation)
                    rows.append({**config.to_dict(), **metrics})
                    rank = (metrics["sharpe"], metrics["total_return"], -metrics["max_drawdown"])
                    best = _select_better_candidate(best, rank, config)

    if best is None:
        raise RuntimeError("Failed to tune Kalman strategy.")

    best_config = best[1]
    spread_full = compute_spread(pair_frame[asset_a], pair_frame[asset_b], hedge_ratio)
    signal_full = build_kalman_signal(spread_full, best_config, observation_var=observation_var)
    full_frame = _evaluate_signal(pair_frame, asset_a, asset_b, hedge_ratio.beta, signal_full, cost_bps)
    split_index = len(train_frame)

    return StrategyResearchResult(
        name="kalman_filtered",
        config=best_config.to_dict(),
        full_frame=full_frame,
        train_metrics=_slice_metrics(full_frame.iloc[:split_index].reset_index(drop=True)),
        test_metrics=_slice_metrics(full_frame.iloc[split_index:].reset_index(drop=True)),
        tuning_grid=pd.DataFrame(rows),
    )


def run_research(config: ResearchConfig, cache_dir: str | None = None) -> ResearchResult:
    pair_frame = load_pair_close_frame(
        asset_a=config.asset_a,
        asset_b=config.asset_b,
        start_date=config.start_date,
        cache_dir=None if cache_dir is None else Path(cache_dir),
    )
    split_index = int(len(pair_frame) * config.train_ratio)
    if split_index <= 1 or split_index >= len(pair_frame):
        raise ValueError("Train ratio must leave at least one observation in both train and test segments.")

    train_frame = pair_frame.iloc[:split_index].reset_index(drop=True)
    asset_a = config.asset_a.upper()
    asset_b = config.asset_b.upper()
    hedge_ratio = estimate_hedge_ratio(train_frame[asset_a], train_frame[asset_b])

    naive_result = _tune_naive_strategy(train_frame, pair_frame, asset_a, asset_b, hedge_ratio, config.cost_bps)
    kalman_result = _tune_kalman_strategy(train_frame, pair_frame, asset_a, asset_b, hedge_ratio, config.cost_bps)

    return ResearchResult(
        config=config,
        hedge_ratio=hedge_ratio,
        pair_frame=pair_frame,
        split_index=split_index,
        naive=naive_result,
        kalman=kalman_result,
    )
