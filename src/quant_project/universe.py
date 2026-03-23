from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from statsmodels.tsa.stattools import coint

from quant_project.backtest import summarize_returns
from quant_project.data import load_pair_close_frame
from quant_project.diagnostics import bootstrap_sharpe_difference
from quant_project.research import BOOTSTRAP_BLOCK_SIZE, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, ResearchConfig, ResearchResult, run_research

CANDIDATE_PAIR_UNIVERSE: list[tuple[str, str]] = [
    ("V", "MA"),
    ("JPM", "BAC"),
    ("XOM", "CVX"),
    ("KO", "PEP"),
    ("HD", "LOW"),
    ("MCD", "YUM"),
    ("TMO", "DHR"),
    ("TLT", "IEF"),
    ("XLF", "VFH"),
    ("XLU", "VPU"),
    ("XLB", "VAW"),
    ("XLI", "VIS"),
    ("GLD", "IAU"),
    ("SPY", "IVV"),
    ("EWA", "EWS"),
    ("EWA", "EWC"),
]


@dataclass(frozen=True)
class UniverseConfig:
    pairs: list[tuple[str, str]]
    start_date: str = "2012-01-01"
    train_ratio: float = 0.6
    cost_bps: float = 5.0
    selection_pvalue_threshold: float | None = 0.1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UniverseResult:
    config: UniverseConfig
    screen_table: pd.DataFrame
    selected_pairs: list[tuple[str, str]]
    pair_results: list[ResearchResult]
    pair_summary: pd.DataFrame
    portfolio_frame: pd.DataFrame
    aggregate_summary: dict[str, Any]


def _screen_candidate_pairs(config: UniverseConfig, cache_dir: str | None = None) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    rows: list[dict[str, Any]] = []
    cache_path = None if cache_dir is None else Path(cache_dir)
    for asset_a, asset_b in config.pairs:
        pair_frame = load_pair_close_frame(asset_a, asset_b, config.start_date, cache_dir=cache_path)
        split_index = int(len(pair_frame) * config.train_ratio)
        train_frame = pair_frame.iloc[:split_index].reset_index(drop=True)
        statistic, pvalue, _ = coint(train_frame[asset_a], train_frame[asset_b])
        rows.append(
            {
                "pair": f"{asset_a}/{asset_b}",
                "asset_a": asset_a,
                "asset_b": asset_b,
                "observations": int(len(pair_frame)),
                "train_engle_granger_stat": float(statistic),
                "train_engle_granger_pvalue": float(pvalue),
            }
        )

    screen_table = pd.DataFrame(rows).sort_values("train_engle_granger_pvalue").reset_index(drop=True)
    if config.selection_pvalue_threshold is None:
        selected_table = screen_table.copy()
    else:
        selected_table = screen_table.loc[
            screen_table["train_engle_granger_pvalue"] <= config.selection_pvalue_threshold
        ].copy()
    selected_pairs = list(zip(selected_table["asset_a"], selected_table["asset_b"], strict=True))
    return screen_table, selected_pairs


def _merge_strategy_returns(pair_results: list[ResearchResult], strategy_name: str) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for result in pair_results:
        strategy = result.naive if strategy_name == "naive" else result.kalman
        pair_name = f"{result.config.asset_a}/{result.config.asset_b}"
        returns_frame = strategy.full_frame.iloc[result.split_index :][["Date", "net_return"]].rename(
            columns={"net_return": pair_name}
        )
        merged = returns_frame if merged is None else merged.merge(returns_frame, on="Date", how="outer")

    if merged is None:
        raise ValueError("No pair results available for portfolio aggregation.")
    return merged.sort_values("Date").reset_index(drop=True)


def _summarize_pair_result(result: ResearchResult) -> dict[str, Any]:
    pair_name = f"{result.config.asset_a}/{result.config.asset_b}"
    diagnostics = result.diagnostics
    return {
        "pair": pair_name,
        "engle_granger_pvalue": diagnostics["engle_granger"]["pvalue"],
        "train_adf_pvalue": diagnostics["spread_stationarity"]["train_adf"]["pvalue"],
        "test_adf_pvalue": diagnostics["spread_stationarity"]["test_adf"]["pvalue"],
        "train_kpss_pvalue": diagnostics["spread_stationarity"]["train_kpss"]["pvalue"],
        "test_kpss_pvalue": diagnostics["spread_stationarity"]["test_kpss"]["pvalue"],
        "static_beta": diagnostics["train_ols"]["beta"],
        "static_beta_ci_low": diagnostics["train_ols"]["beta_ci_95"][0],
        "static_beta_ci_high": diagnostics["train_ols"]["beta_ci_95"][1],
        "naive_test_sharpe": result.naive.test_metrics["sharpe"],
        "naive_test_total_return": result.naive.test_metrics["total_return"],
        "kalman_test_sharpe": result.kalman.test_metrics["sharpe"],
        "kalman_test_total_return": result.kalman.test_metrics["total_return"],
        "kalman_minus_naive_sharpe": result.kalman.test_metrics["sharpe"] - result.naive.test_metrics["sharpe"],
        "kalman_minus_naive_total_return": result.kalman.test_metrics["total_return"] - result.naive.test_metrics["total_return"],
        "kalman_outperforms_on_sharpe": float(result.kalman.test_metrics["sharpe"] > result.naive.test_metrics["sharpe"]),
        "kalman_outperforms_on_total_return": float(
            result.kalman.test_metrics["total_return"] > result.naive.test_metrics["total_return"]
        ),
    }


def run_universe_research(config: UniverseConfig, cache_dir: str | None = None) -> UniverseResult:
    screen_table, selected_pairs = _screen_candidate_pairs(config, cache_dir=cache_dir)
    if not selected_pairs:
        raise ValueError("The pair-selection rule removed every candidate pair.")

    pair_results: list[ResearchResult] = []
    for asset_a, asset_b in selected_pairs:
        pair_results.append(
            run_research(
                ResearchConfig(
                    asset_a=asset_a,
                    asset_b=asset_b,
                    start_date=config.start_date,
                    train_ratio=config.train_ratio,
                    cost_bps=config.cost_bps,
                ),
                cache_dir=cache_dir,
            )
        )

    pair_summary = pd.DataFrame([_summarize_pair_result(result) for result in pair_results]).sort_values(
        "kalman_minus_naive_sharpe",
        ascending=False,
    )
    naive_returns = _merge_strategy_returns(pair_results, "naive")
    kalman_returns = _merge_strategy_returns(pair_results, "kalman")

    portfolio_frame = naive_returns[["Date"]].copy()
    portfolio_frame["naive_return"] = naive_returns.drop(columns=["Date"]).mean(axis=1, skipna=True)
    portfolio_frame["kalman_return"] = kalman_returns.drop(columns=["Date"]).mean(axis=1, skipna=True)
    portfolio_frame["naive_equity"] = (1.0 + portfolio_frame["naive_return"].fillna(0.0)).cumprod()
    portfolio_frame["kalman_equity"] = (1.0 + portfolio_frame["kalman_return"].fillna(0.0)).cumprod()

    naive_metrics = summarize_returns(portfolio_frame["naive_return"])
    kalman_metrics = summarize_returns(portfolio_frame["kalman_return"])
    aggregate_summary = {
        "candidate_pair_count": int(len(config.pairs)),
        "selected_pair_count": int(len(selected_pairs)),
        "selection_pvalue_threshold": config.selection_pvalue_threshold,
        "selected_pairs": [f"{asset_a}/{asset_b}" for asset_a, asset_b in selected_pairs],
        "cointegrated_pairs_at_5pct": int((pair_summary["engle_granger_pvalue"] < 0.05).sum()),
        "kalman_beats_naive_on_test_sharpe": int(pair_summary["kalman_outperforms_on_sharpe"].sum()),
        "kalman_beats_naive_on_test_total_return": int(pair_summary["kalman_outperforms_on_total_return"].sum()),
        "mean_pair_test_sharpe_delta": float(pair_summary["kalman_minus_naive_sharpe"].mean()),
        "median_pair_test_sharpe_delta": float(pair_summary["kalman_minus_naive_sharpe"].median()),
        "portfolio_naive": naive_metrics,
        "portfolio_kalman": kalman_metrics,
        "portfolio_sharpe_delta_bootstrap": bootstrap_sharpe_difference(
            portfolio_frame["kalman_return"],
            portfolio_frame["naive_return"],
            block_size=BOOTSTRAP_BLOCK_SIZE,
            n_boot=BOOTSTRAP_SAMPLES,
            seed=BOOTSTRAP_SEED,
        ),
    }

    return UniverseResult(
        config=config,
        screen_table=screen_table,
        selected_pairs=selected_pairs,
        pair_results=pair_results,
        pair_summary=pair_summary.reset_index(drop=True),
        portfolio_frame=portfolio_frame,
        aggregate_summary=aggregate_summary,
    )
