from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quant_project.plots import plot_equity_curves, plot_kalman_stability, plot_signal_vs_spread
from quant_project.research import ResearchConfig, run_research


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-a", default="V")
    parser.add_argument("--asset-b", default="MA")
    parser.add_argument("--start-date", default="2012-01-01")
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--output-dir", default="results/v_ma")
    parser.add_argument("--cache-dir", default="data/cache")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    research_result = run_research(
        ResearchConfig(
            asset_a=args.asset_a.upper(),
            asset_b=args.asset_b.upper(),
            start_date=args.start_date,
            train_ratio=args.train_ratio,
            cost_bps=args.cost_bps,
        ),
        cache_dir=args.cache_dir,
    )

    summary = research_result.to_summary_dict()

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    research_result.naive.full_frame.to_csv(output_dir / "naive_daily_frame.csv", index=False)
    research_result.kalman.full_frame.to_csv(output_dir / "kalman_daily_frame.csv", index=False)
    research_result.naive.tuning_grid.to_csv(output_dir / "naive_tuning_grid.csv", index=False)
    research_result.kalman.tuning_grid.to_csv(output_dir / "kalman_tuning_grid.csv", index=False)

    plot_signal_vs_spread(
        strategy_frame=research_result.kalman.full_frame.iloc[research_result.split_index :].reset_index(drop=True),
        title=f"{args.asset_a.upper()}/{args.asset_b.upper()} spread vs Kalman equilibrium (test set)",
        entry_z=float(research_result.kalman.config["entry_z"]),
        exit_z=float(research_result.kalman.config["exit_z"]),
        output_path=output_dir / "signal_vs_spread.png",
    )
    plot_equity_curves(research_result, output_dir / "equity_curves.png")
    plot_kalman_stability(
        tuning_grid=research_result.kalman.tuning_grid,
        selected_exit_z=float(research_result.kalman.config["exit_z"]),
        selected_vol_span=int(research_result.kalman.config["vol_span"]),
        output_path=output_dir / "kalman_stability.png",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
