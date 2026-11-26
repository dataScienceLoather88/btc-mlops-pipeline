"""ETL step 1: fetch daily BTC-USD OHLCV data and store it as raw CSV."""

import argparse
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", "data/raw/btc_raw.csv")
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


# fetch data from yfinance api helper function
def _fetch_yfinance(days: int) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(
        "BTC-USD",
        period=f"{days}d",
        interval="1d",
        progress=False,
        auto_adjust=True,
    )
    if df is None or df.empty:
        raise RuntimeError("yfinance returned no data")

    # Newer yfinance versions return MultiIndex columns (Price, Ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[OHLCV].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    return df.astype(float)


# synthetically generate dataset
# only use when api fails/network is down
def _synthetic_fallback(days: int) -> pd.DataFrame:
    """Deterministic random-walk OHLCV, used only if the network is down."""
    rng = np.random.default_rng(42)
    idx = pd.date_range(end=datetime.now(timezone.utc).date(), periods=days, freq="D")
    close = 60000 + np.cumsum(rng.normal(100, 800, days))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.02, days))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.02, days))
    volume = rng.uniform(1e10, 3e10, days)
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )
    df.index.name = "Date"
    return df


# return structured btc dataset/dataframe
# (either synthetically generated data or the yfinance API).
def fetch_btc_ohlcv(days: int = 730, allow_fallback: bool = True):
    """Return (dataframe, source) where source is 'yfinance' or 'synthetic'."""
    try:
        df = _fetch_yfinance(days)
        source = "yfinance"
    except Exception as exc:
        if not allow_fallback:
            raise
        print(
            f"[etl_fetch] WARNING: live fetch failed ({exc}); " "using synthetic data"
        )
        df = _synthetic_fallback(days)
        source = "synthetic"

    # Today's candle is incomplete (BTC trades 24/7), so keep closed days only
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    df = df[df.index < today]
    return df, source


# store resulting dataframe as raw csv within directory /data/raw/
def run_fetch(output_path: str = RAW_DATA_PATH, days: int = 730) -> str:
    df, source = fetch_btc_ohlcv(days=days)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path)
    print(
        f"[etl_fetch] source={source} rows={len(df)} "
        f"range={df.index.min().date()}..{df.index.max().date()} -> {output_path}"
    )
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch BTC-USD daily data")
    parser.add_argument("--output", default=RAW_DATA_PATH)
    parser.add_argument("--days", type=int, default=730)
    args = parser.parse_args()
    run_fetch(args.output, args.days)
