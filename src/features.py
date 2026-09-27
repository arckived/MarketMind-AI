from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _safe_pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    return series.pct_change(periods=periods).replace([np.inf, -np.inf], np.nan)


def build_feature_table(
    stock: pd.DataFrame,
    benchmark: pd.DataFrame | None = None,
    vix: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create time-series features for market regime detection.

    The output contains both raw chart columns and model-ready features.
    """
    price = stock["Adj Close"].astype(float).rename("price")
    close = stock["Close"].astype(float).rename("close")
    volume = stock["Volume"].astype(float).rename("volume")

    df = pd.DataFrame(index=stock.index)
    df["price"] = price
    df["close"] = close
    df["volume"] = volume

    # Returns and trend features
    df["ret_1d"] = _safe_pct_change(price, 1)
    df["log_ret_1d"] = np.log(price / price.shift(1)).replace([np.inf, -np.inf], np.nan)
    df["ret_5d"] = _safe_pct_change(price, 5)
    df["ret_20d"] = _safe_pct_change(price, 20)
    df["ret_60d"] = _safe_pct_change(price, 60)
    df["ret_120d"] = _safe_pct_change(price, 120)

    # Volatility features
    df["vol_20d"] = df["ret_1d"].rolling(20).std() * np.sqrt(TRADING_DAYS)
    df["vol_60d"] = df["ret_1d"].rolling(60).std() * np.sqrt(TRADING_DAYS)
    df["vol_ratio_20_60"] = df["vol_20d"] / df["vol_60d"]

    # Moving averages and gap features
    for window in [20, 50, 100, 200]:
        df[f"ma_{window}"] = price.rolling(window).mean()
        df[f"price_ma_{window}_gap"] = (price / df[f"ma_{window}"]) - 1

    # Momentum features
    df["momentum_20d"] = price / price.shift(20) - 1
    df["momentum_60d"] = price / price.shift(60) - 1
    df["momentum_120d"] = price / price.shift(120) - 1

    # Risk features
    running_max = price.cummax()
    df["drawdown"] = (price / running_max) - 1
    df["drawdown_60d_min"] = df["drawdown"].rolling(60).min()

    # Volume features
    df["volume_change_20d"] = volume.pct_change(20).replace([np.inf, -np.inf], np.nan)
    df["volume_ma_20"] = volume.rolling(20).mean()
    volume_std_20 = volume.rolling(20).std()
    df["volume_zscore_20"] = ((volume - df["volume_ma_20"]) / volume_std_20).replace([np.inf, -np.inf], np.nan)

    # Benchmark features
    if benchmark is not None and not benchmark.empty:
        bench_price = benchmark["Adj Close"].astype(float).reindex(df.index).ffill()
        df["benchmark_price"] = bench_price
        df["benchmark_ret_1d"] = _safe_pct_change(bench_price, 1)
        df["benchmark_ret_20d"] = _safe_pct_change(bench_price, 20)
        df["benchmark_vol_20d"] = df["benchmark_ret_1d"].rolling(20).std() * np.sqrt(TRADING_DAYS)
        df["excess_ret_20d"] = df["ret_20d"] - df["benchmark_ret_20d"]
        df["corr_with_benchmark_60d"] = df["ret_1d"].rolling(60).corr(df["benchmark_ret_1d"])
    else:
        for col in [
            "benchmark_price",
            "benchmark_ret_1d",
            "benchmark_ret_20d",
            "benchmark_vol_20d",
            "excess_ret_20d",
            "corr_with_benchmark_60d",
        ]:
            df[col] = np.nan

    # VIX features
    if vix is not None and not vix.empty:
        vix_price = vix["Adj Close"].astype(float).reindex(df.index).ffill()
        df["vix_level"] = vix_price
        df["vix_change_20d"] = _safe_pct_change(vix_price, 20)
        df["vix_ma_20"] = vix_price.rolling(20).mean()
    else:
        for col in ["vix_level", "vix_change_20d", "vix_ma_20"]:
            df[col] = np.nan

    # Clean features, but keep enough rows for charts.
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


MODEL_FEATURES = [
    "ret_20d",
    "ret_60d",
    "ret_120d",
    "vol_20d",
    "vol_60d",
    "vol_ratio_20_60",
    "price_ma_50_gap",
    "price_ma_200_gap",
    "momentum_20d",
    "momentum_60d",
    "momentum_120d",
    "drawdown",
    "drawdown_60d_min",
    "volume_change_20d",
    "volume_zscore_20",
    "benchmark_ret_20d",
    "benchmark_vol_20d",
    "excess_ret_20d",
    "corr_with_benchmark_60d",
    "vix_level",
    "vix_change_20d",
]


def get_model_matrix(features: pd.DataFrame, feature_cols: list[str] | None = None) -> pd.DataFrame:
    """Return a cleaned model matrix with available feature columns."""
    cols = feature_cols or MODEL_FEATURES
    available_cols = [col for col in cols if col in features.columns]
    matrix = features[available_cols].copy()
    # Fill isolated missing macro values without leaking future labels.
    matrix = matrix.ffill().bfill()
    return matrix.dropna()
