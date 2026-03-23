from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

cache_root = Path(tempfile.gettempdir()) / "quant_project_cache"
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from quant_project.research import ResearchResult


def _save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_signal_vs_spread(
    strategy_frame: pd.DataFrame,
    title: str,
    entry_z: float,
    exit_z: float,
    output_path: Path,
    spread_label: str = "Spread residual",
    anchor_label: str = "Equilibrium",
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    axes[0].plot(strategy_frame["Date"], strategy_frame["spread"], label=spread_label, linewidth=1.2, color="#1b4965")
    anchor = strategy_frame["anchor"]
    if float(anchor.abs().max()) < 1e-12:
        axes[0].axhline(0.0, label=anchor_label, linewidth=1.2, color="#ca6702")
    else:
        axes[0].plot(strategy_frame["Date"], anchor, label=anchor_label, linewidth=1.2, color="#ca6702")
    axes[0].set_ylabel("Residual")
    axes[0].set_title(title)
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.25)

    axes[1].plot(strategy_frame["Date"], strategy_frame["zscore"], color="#005f73", linewidth=1.0)
    axes[1].axhline(entry_z, color="#bb3e03", linestyle="--", linewidth=1.0, label="Entry")
    axes[1].axhline(-entry_z, color="#bb3e03", linestyle="--", linewidth=1.0)
    axes[1].axhline(exit_z, color="#94d2bd", linestyle=":", linewidth=1.0, label="Exit")
    axes[1].axhline(-exit_z, color="#94d2bd", linestyle=":", linewidth=1.0)
    axes[1].fill_between(strategy_frame["Date"], -entry_z, entry_z, color="#e9d8a6", alpha=0.2)
    axes[1].set_ylabel("Z-score")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.25)

    _save_figure(output_path)


def plot_equity_curves(research_result: ResearchResult, output_path: Path) -> None:
    split_date = research_result.pair_frame.loc[research_result.split_index, "Date"]
    fig, axis = plt.subplots(figsize=(12, 5))

    axis.plot(
        research_result.naive.full_frame["Date"],
        research_result.naive.full_frame["equity"],
        label=research_result.naive.label,
        linewidth=1.5,
        color="#5f0f40",
    )
    axis.plot(
        research_result.kalman.full_frame["Date"],
        research_result.kalman.full_frame["equity"],
        label=research_result.kalman.label,
        linewidth=1.5,
        color="#0a9396",
    )
    axis.axvline(split_date, color="#6c757d", linestyle="--", linewidth=1.1, label="Train / test split")
    axis.set_title("Equity curves with transaction costs")
    axis.set_ylabel("Equity")
    axis.legend(loc="upper left")
    axis.grid(alpha=0.25)

    _save_figure(output_path)


def plot_kalman_stability(
    tuning_grid: pd.DataFrame,
    selected_exit_z: float,
    selected_vol_span: int,
    output_path: Path,
) -> None:
    subset = tuning_grid[(tuning_grid["exit_z"] == selected_exit_z) & (tuning_grid["vol_span"] == selected_vol_span)].copy()
    pivot = subset.pivot_table(
        index="process_var_multiplier",
        columns="entry_z",
        values="sharpe",
        aggfunc="mean",
    ).sort_index()

    fig, axis = plt.subplots(figsize=(8, 5))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="RdYlGn")
    axis.set_xticks(range(len(pivot.columns)))
    axis.set_xticklabels([f"{value:.2f}" for value in pivot.columns])
    axis.set_yticks(range(len(pivot.index)))
    axis.set_yticklabels([f"{value:.0e}" for value in pivot.index])
    axis.set_xlabel("Entry z-score")
    axis.set_ylabel("State noise multiplier")
    axis.set_title("Kalman regression train-set Sharpe stability")

    for row_idx, process_var in enumerate(pivot.index):
        for col_idx, entry in enumerate(pivot.columns):
            axis.text(
                col_idx,
                row_idx,
                f"{pivot.loc[process_var, entry]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    fig.colorbar(image, ax=axis, shrink=0.85, label="Sharpe")
    _save_figure(output_path)


def plot_hedge_ratio_path(strategy_frame: pd.DataFrame, split_date: pd.Timestamp, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(12, 4))
    axis.plot(strategy_frame["Date"], strategy_frame["beta"], linewidth=1.4, color="#386641", label="Dynamic beta_t")
    axis.axvline(split_date, color="#6c757d", linestyle="--", linewidth=1.1, label="Train / test split")
    axis.set_title("Time-varying hedge ratio from Kalman regression")
    axis.set_ylabel("Beta")
    axis.legend(loc="upper left")
    axis.grid(alpha=0.25)
    _save_figure(output_path)


def plot_universe_test_sharpes(pair_summary: pd.DataFrame, output_path: Path) -> None:
    labels = pair_summary["pair"].tolist()
    positions = np.arange(len(labels))
    width = 0.36

    fig, axis = plt.subplots(figsize=(12, 5))
    axis.bar(positions - width / 2, pair_summary["naive_test_sharpe"], width=width, color="#8d99ae", label="Naive")
    axis.bar(
        positions + width / 2,
        pair_summary["kalman_test_sharpe"],
        width=width,
        color="#2a9d8f",
        label="Kalman state-space",
    )
    axis.axhline(0.0, color="#6c757d", linewidth=1.0)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_ylabel("Test Sharpe")
    axis.set_title("Out-of-sample Sharpe by pair")
    axis.legend(loc="upper left")
    axis.grid(alpha=0.2, axis="y")
    _save_figure(output_path)


def plot_universe_portfolio_equity(portfolio_frame: pd.DataFrame, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(12, 5))
    axis.plot(portfolio_frame["Date"], portfolio_frame["naive_equity"], color="#8d99ae", linewidth=1.5, label="Naive")
    axis.plot(
        portfolio_frame["Date"],
        portfolio_frame["kalman_equity"],
        color="#2a9d8f",
        linewidth=1.5,
        label="Kalman state-space",
    )
    axis.set_title("Equal-weight test portfolio across the pair universe")
    axis.set_ylabel("Equity")
    axis.legend(loc="upper left")
    axis.grid(alpha=0.25)
    _save_figure(output_path)
