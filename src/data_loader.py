from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class MarketDataBundle:
    ticker: str
    benchmark: str
    vix_symbol: str
    stock: pd.DataFrame
    benchmark_df: pd.DataFrame
    vix: pd.DataFrame


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make yfinance output consistent across versions/tickers."""
    if isinstance(df.columns, pd.MultiIndex):
        # When a single ticker is downloaded, yfinance may return (Price, Ticker).
        # Keep the first level when it contains standard OHLCV names.
        first_level = list(df.columns.get_level_values(0))
        second_level = list(df.columns.get_level_values(1))
        standard = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        if any(col in standard for col in first_level):
            df = df.copy()
            df.columns = df.columns.get_level_values(0)
        else:
            df = df.copy()
            df.columns = df.columns.get_level_values(-1)
    return df


def _download_one(ticker: str, start: str, end: Optional[str] = None) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    df = _flatten_columns(df)
    if df.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    # Some newer yfinance configs may omit Adj Close. Fall back safely.
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]
    expected = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in expected:
        if col not in df.columns:
            df[col] = pd.NA
    return df[expected].dropna(subset=["Adj Close"])


def load_market_data(
    ticker: str = "AAPL",
    benchmark: str = "SPY",
    vix_symbol: str = "^VIX",
    start: str = "1995-01-01",
    end: Optional[str] = None,
) -> MarketDataBundle:
    """Download stock, benchmark, and volatility index data."""
    ticker = ticker.upper().strip()
    benchmark = benchmark.upper().strip()
    stock = _download_one(ticker, start, end)
    benchmark_df = _download_one(benchmark, start, end)
    vix = _download_one(vix_symbol, start, end)
    return MarketDataBundle(
        ticker=ticker,
        benchmark=benchmark,
        vix_symbol=vix_symbol,
        stock=stock,
        benchmark_df=benchmark_df,
        vix=vix,
    )
