from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from quant_project.backtest import BacktestConfig, backtest_pair, summarize_returns
from quant_project.data import load_pair_close_frame
from quant_project.diagnostics import bootstrap_sharpe_ci, bootstrap_sharpe_difference, run_pair_diagnostics
from quant_project.signals import (
    HedgeRatio,
    KalmanRegressionSignalConfig,
    NaiveSignalConfig,
    SignalFrame,
    build_kalman_regression_signal,
    build_naive_signal,
    compute_spread,
    estimate_hedge_ratio,
)

BOOTSTRAP_BLOCK_SIZE = 20
BOOTSTRAP_SAMPLES = 300
BOOTSTRAP_SEED = 7


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
    label: str
    config: dict[str, float | int]
    full_frame: pd.DataFrame
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    tuning_grid: pd.DataFrame
    test_uncertainty: dict[str, float | int]


@dataclass
class ResearchResult:
    config: ResearchConfig
    hedge_ratio: HedgeRatio
    pair_frame: pd.DataFrame
    split_index: int
    diagnostics: dict[str, Any]
    naive: StrategyResearchResult
    kalman: StrategyResearchResult
    comparison_uncertainty: dict[str, float | int]

    def to_summary_dict(self) -> dict[str, Any]:
        split_date = self.pair_frame.loc[self.split_index, "Date"].strftime("%Y-%m-%d")
        return {
            "research_config": self.config.to_dict(),
            "hedge_ratio": self.hedge_ratio.to_dict(),
            "split_index": self.split_index,
            "train_end_date": self.pair_frame.loc[self.split_index - 1, "Date"].strftime("%Y-%m-%d"),
            "test_start_date": split_date,
            "diagnostics": self.diagnostics,
            "naive": {
                "label": self.naive.label,
                "config": self.naive.config,
                "train_metrics": self.naive.train_metrics,
                "test_metrics": self.naive.test_metrics,
                "test_uncertainty": self.naive.test_uncertainty,
            },
            "kalman": {
                "label": self.kalman.label,
                "config": self.kalman.config,
                "train_metrics": self.kalman.train_metrics,
                "test_metrics": self.kalman.test_metrics,
                "test_uncertainty": self.kalman.test_uncertainty,
            },
            "strategy_difference": {
                "test_sharpe_delta": self.kalman.test_metrics["sharpe"] - self.naive.test_metrics["sharpe"],
                "test_total_return_delta": self.kalman.test_metrics["total_return"] - self.naive.test_metrics["total_return"],
                "bootstrap_sharpe_delta": self.comparison_uncertainty,
            },
        }


def _extract_strategy_beta(strategy_result: StrategyResearchResult, fallback_beta: float) -> float | pd.Series:
    return strategy_result.full_frame["beta"] if "beta" in strategy_result.full_frame.columns else fallback_beta


def build_cost_sensitivity_table(
    research_result: ResearchResult,
    cost_grid: list[float] | tuple[float, ...] = (0.0, 2.0, 5.0, 10.0),
) -> pd.DataFrame:
    asset_a = research_result.config.asset_a
    asset_b = research_result.config.asset_b
    price_frame = research_result.pair_frame.loc[:, ["Date", asset_a, asset_b]]
    rows: list[dict[str, float | str]] = []

    for strategy_name, strategy_result in [("naive", research_result.naive), ("kalman", research_result.kalman)]:
        beta_input = _extract_strategy_beta(strategy_result, research_result.hedge_ratio.beta)
        positions = strategy_result.full_frame["position"]
        for cost_bps in cost_grid:
            backtest = backtest_pair(
                prices=price_frame,
                asset_a=asset_a,
                asset_b=asset_b,
                beta=beta_input,
                positions=positions,
                config=BacktestConfig(cost_bps=cost_bps),
            )
            test_frame = backtest.frame.iloc[research_result.split_index :].reset_index(drop=True)
            metrics = summarize_returns(
                test_frame["net_return"],
                turnover=test_frame["turnover"],
                notional_turnover=test_frame["notional_turnover"],
            )
            rows.append(
                {
                    "strategy": strategy_name,
                    "label": strategy_result.label,
                    "cost_bps": float(cost_bps),
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


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
    fallback_beta: float,
    signal: SignalFrame,
    cost_bps: float,
) -> pd.DataFrame:
    beta_input: float | pd.Series = signal.frame["beta"] if "beta" in signal.frame.columns else fallback_beta
    backtest = backtest_pair(
        prices=pair_frame.loc[:, ["Date", asset_a, asset_b]],
        asset_a=asset_a,
        asset_b=asset_b,
        beta=beta_input,
        positions=signal.frame["position"],
        config=BacktestConfig(cost_bps=cost_bps),
    )
    enriched = pair_frame.copy()
    for column in signal.frame.columns:
        enriched[column] = signal.frame[column].to_numpy()
    for column in ["gross_return", "transaction_cost", "net_return", "equity", "turnover", "notional_turnover"]:
        enriched[column] = backtest.frame[column].to_numpy()
    if "beta" not in enriched.columns:
        enriched["beta"] = fallback_beta
    return enriched


def _slice_metrics(frame: pd.DataFrame) -> dict[str, float]:
    return summarize_returns(
        frame["net_return"],
        turnover=frame["turnover"],
        notional_turnover=frame["notional_turnover"],
    )


def _make_uncertainty_summary(frame: pd.DataFrame) -> dict[str, float | int]:
    return bootstrap_sharpe_ci(
        frame["net_return"],
        block_size=BOOTSTRAP_BLOCK_SIZE,
        n_boot=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )


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
    test_frame = full_frame.iloc[split_index:].reset_index(drop=True)

    return StrategyResearchResult(
        name="rolling_zscore",
        label="Static hedge ratio + rolling z-score",
        config=best_config.to_dict(),
        full_frame=full_frame,
        train_metrics=_slice_metrics(full_frame.iloc[:split_index].reset_index(drop=True)),
        test_metrics=_slice_metrics(test_frame),
        tuning_grid=pd.DataFrame(rows),
        test_uncertainty=_make_uncertainty_summary(test_frame),
    )


def _tune_kalman_strategy(
    train_frame: pd.DataFrame,
    pair_frame: pd.DataFrame,
    asset_a: str,
    asset_b: str,
    hedge_ratio: HedgeRatio,
    cost_bps: float,
) -> StrategyResearchResult:
    train_size = len(train_frame)
    best: tuple[tuple[float, float, float], KalmanRegressionSignalConfig] | None = None
    rows: list[dict[str, float | int]] = []

    for process_var_multiplier in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]:
        for entry_z in [1.0, 1.25, 1.5, 1.75, 2.0]:
            for exit_z in [0.0, 0.25, 0.5, 0.75]:
                if exit_z >= entry_z:
                    continue
                for vol_span in [10, 20, 40, 60]:
                    config = KalmanRegressionSignalConfig(
                        process_var_multiplier=process_var_multiplier,
                        entry_z=entry_z,
                        exit_z=exit_z,
                        vol_span=vol_span,
                    )
                    signal_train = build_kalman_regression_signal(
                        train_frame[asset_a],
                        train_frame[asset_b],
                        train_size=train_size,
                        config=config,
                    )
                    evaluation = _evaluate_signal(train_frame, asset_a, asset_b, hedge_ratio.beta, signal_train, cost_bps)
                    metrics = _slice_metrics(evaluation)
                    rows.append({**config.to_dict(), **metrics})
                    rank = (metrics["sharpe"], metrics["total_return"], -metrics["max_drawdown"])
                    best = _select_better_candidate(best, rank, config)

    if best is None:
        raise RuntimeError("Failed to tune Kalman regression strategy.")

    best_config = best[1]
    signal_full = build_kalman_regression_signal(
        pair_frame[asset_a],
        pair_frame[asset_b],
        train_size=train_size,
        config=best_config,
    )
    full_frame = _evaluate_signal(pair_frame, asset_a, asset_b, hedge_ratio.beta, signal_full, cost_bps)
    split_index = train_size
    test_frame = full_frame.iloc[split_index:].reset_index(drop=True)

    return StrategyResearchResult(
        name="kalman_regression",
        label="Kalman state-space regression",
        config=best_config.to_dict(),
        full_frame=full_frame,
        train_metrics=_slice_metrics(full_frame.iloc[:split_index].reset_index(drop=True)),
        test_metrics=_slice_metrics(test_frame),
        tuning_grid=pd.DataFrame(rows),
        test_uncertainty=_make_uncertainty_summary(test_frame),
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
    test_frame = pair_frame.iloc[split_index:].reset_index(drop=True)
    asset_a = config.asset_a.upper()
    asset_b = config.asset_b.upper()
    hedge_ratio = estimate_hedge_ratio(train_frame[asset_a], train_frame[asset_b])
    diagnostics = run_pair_diagnostics(train_frame, test_frame, asset_a, asset_b)

    naive_result = _tune_naive_strategy(train_frame, pair_frame, asset_a, asset_b, hedge_ratio, config.cost_bps)
    kalman_result = _tune_kalman_strategy(train_frame, pair_frame, asset_a, asset_b, hedge_ratio, config.cost_bps)
    comparison_uncertainty = bootstrap_sharpe_difference(
        kalman_result.full_frame.iloc[split_index:]["net_return"],
        naive_result.full_frame.iloc[split_index:]["net_return"],
        block_size=BOOTSTRAP_BLOCK_SIZE,
        n_boot=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )

    return ResearchResult(
        config=config,
        hedge_ratio=hedge_ratio,
        pair_frame=pair_frame,
        split_index=split_index,
        diagnostics=diagnostics,
        naive=naive_result,
        kalman=kalman_result,
        comparison_uncertainty=comparison_uncertainty,
    )
