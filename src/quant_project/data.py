from __future__ import annotations

from pathlib import Path

import pandas as pd

STOOQ_DAILY_CLOSE_URL = "https://stooq.com/q/d/l/?s={symbol}.us&i=d"


def fetch_stooq_close(symbol: str, cache_dir: Path | None = None) -> pd.DataFrame:
    """Load daily close prices for a US-listed ticker from Stooq."""
    normalized = symbol.strip().lower()
    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{normalized}.csv"

    if cache_path is not None and cache_path.exists():
        frame = pd.read_csv(cache_path, parse_dates=["Date"])
    else:
        frame = pd.read_csv(STOOQ_DAILY_CLOSE_URL.format(symbol=normalized), parse_dates=["Date"])
        if cache_path is not None:
            frame.to_csv(cache_path, index=False)

    output = (
        frame.loc[:, ["Date", "Close"]]
        .dropna()
        .sort_values("Date")
        .rename(columns={"Close": symbol.upper()})
        .reset_index(drop=True)
    )
    return output


def load_pair_close_frame(
    asset_a: str,
    asset_b: str,
    start_date: str,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Download, align, and trim the close series for a candidate pair."""
    asset_a = asset_a.upper()
    asset_b = asset_b.upper()
    left = fetch_stooq_close(asset_a, cache_dir=cache_dir)
    right = fetch_stooq_close(asset_b, cache_dir=cache_dir)

    pair = (
        left.merge(right, on="Date", how="inner")
        .sort_values("Date")
        .query("Date >= @start_date")
        .dropna()
        .reset_index(drop=True)
    )

    if pair.empty:
        raise ValueError(f"No overlapping prices found for {asset_a}/{asset_b} after {start_date}.")

    return pair
