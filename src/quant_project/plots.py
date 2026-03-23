from __future__ import annotations

import os
import tempfile
from pathlib import Path

cache_root = Path(tempfile.gettempdir()) / "quant_project_cache"
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    axes[0].plot(strategy_frame["Date"], strategy_frame["spread"], label="Observed spread", linewidth=1.2, color="#1b4965")
    axes[0].plot(strategy_frame["Date"], strategy_frame["anchor"], label="Filtered equilibrium", linewidth=1.2, color="#ca6702")
    axes[0].set_ylabel("Spread")
    axes[0].set_title(title)
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.25)

    axes[1].plot(strategy_frame["Date"], strategy_frame["zscore"], color="#005f73", linewidth=1.0)
    axes[1].axhline(entry_z, color="#bb3e03", linestyle="--", linewidth=1.0, label="Entry")
    axes[1].axhline(-entry_z, color="#bb3e03", linestyle="--", linewidth=1.0)
    axes[1].axhline(exit_z, color="#94d2bd", linestyle=":", linewidth=1.0, label="Exit")
    axes[1].axhline(-exit_z, color="#94d2bd", linestyle=":", linewidth=1.0)
    axes[1].fill_between(
        strategy_frame["Date"],
        -entry_z,
        entry_z,
        color="#e9d8a6",
        alpha=0.2,
    )
    axes[1].set_ylabel("Z-score")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.25)

    _save_figure(output_path)


def plot_equity_curves(
    research_result: ResearchResult,
    output_path: Path,
) -> None:
    split_date = research_result.pair_frame.loc[research_result.split_index, "Date"]
    fig, axis = plt.subplots(figsize=(12, 5))

    axis.plot(
        research_result.naive.full_frame["Date"],
        research_result.naive.full_frame["equity"],
        label="Naive rolling z-score",
        linewidth=1.5,
        color="#5f0f40",
    )
    axis.plot(
        research_result.kalman.full_frame["Date"],
        research_result.kalman.full_frame["equity"],
        label="Kalman filtered signal",
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
    subset = tuning_grid[
        (tuning_grid["exit_z"] == selected_exit_z) & (tuning_grid["vol_span"] == selected_vol_span)
    ].copy()
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
    axis.set_ylabel("Process variance multiplier")
    axis.set_title("Kalman train-set Sharpe stability")

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
