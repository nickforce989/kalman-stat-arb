from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quant_project.plots import plot_universe_portfolio_equity, plot_universe_test_sharpes
from quant_project.universe import CANDIDATE_PAIR_UNIVERSE, UniverseConfig, run_universe_research


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2012-01-01")
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--selection-pvalue-threshold", type=float, default=0.1)
    parser.add_argument("--output-dir", default="results/universe")
    parser.add_argument("--cache-dir", default="data/cache")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    universe_result = run_universe_research(
        UniverseConfig(
            pairs=CANDIDATE_PAIR_UNIVERSE,
            start_date=args.start_date,
            train_ratio=args.train_ratio,
            cost_bps=args.cost_bps,
            selection_pvalue_threshold=args.selection_pvalue_threshold,
        ),
        cache_dir=args.cache_dir,
    )

    pair_summary_path = output_dir / "pair_summary.csv"
    screen_table_path = output_dir / "screen_table.csv"
    aggregate_summary_path = output_dir / "aggregate_summary.json"
    portfolio_returns_path = output_dir / "portfolio_returns.csv"

    universe_result.screen_table.to_csv(screen_table_path, index=False)
    universe_result.pair_summary.to_csv(pair_summary_path, index=False)
    universe_result.portfolio_frame.to_csv(portfolio_returns_path, index=False)
    aggregate_summary_path.write_text(json.dumps(universe_result.aggregate_summary, indent=2), encoding="utf-8")

    plot_universe_test_sharpes(universe_result.pair_summary, output_dir / "pair_test_sharpes.png")
    plot_universe_portfolio_equity(universe_result.portfolio_frame, output_dir / "portfolio_equity.png")

    print(json.dumps(universe_result.aggregate_summary, indent=2))


if __name__ == "__main__":
    main()
