from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.features import MODEL_FEATURES, get_model_matrix


REGIME_COLORS = {
    "Bullish": "#2ca02c",
    "Bearish": "#d62728",
    "Sideways": "#7f7f7f",
    "High Volatility": "#ff7f0e",
    "Low Volatility": "#1f77b4",
}


@dataclass
class RegimeResult:
    data: pd.DataFrame
    model: object
    scaler: StandardScaler
    feature_cols: list[str]
    cluster_to_regime: dict[int, str]
    cluster_stats: pd.DataFrame


def _map_clusters_to_regimes(cluster_stats: pd.DataFrame) -> dict[int, str]:
    """Map unsupervised clusters to readable finance labels.

    This is intentionally rule-based so the labels are explainable.
    """
    remaining = set(cluster_stats.index.tolist())
    mapping: dict[int, str] = {}

    def pick(metric: str, mode: str = "max") -> int | None:
        candidates = [idx for idx in cluster_stats.index if idx in remaining]
        if not candidates:
            return None
        values = cluster_stats.loc[candidates, metric]
        return int(values.idxmax() if mode == "max" else values.idxmin())

    high_vol = pick("vol_20d", "max")
    if high_vol is not None:
        mapping[high_vol] = "High Volatility"
        remaining.discard(high_vol)

    bear_score = cluster_stats["ret_60d"] + cluster_stats["price_ma_200_gap"]
    if remaining:
        bear = int(bear_score.loc[list(remaining)].idxmin())
        mapping[bear] = "Bearish"
        remaining.discard(bear)

    bull_score = cluster_stats["ret_60d"] + cluster_stats["price_ma_200_gap"] + cluster_stats["excess_ret_20d"].fillna(0)
    if remaining:
        bull = int(bull_score.loc[list(remaining)].idxmax())
        mapping[bull] = "Bullish"
        remaining.discard(bull)

    low_vol = pick("vol_20d", "min")
    if low_vol is not None and low_vol in remaining:
        mapping[low_vol] = "Low Volatility"
        remaining.discard(low_vol)

    for cluster_id in remaining:
        mapping[int(cluster_id)] = "Sideways"

    return mapping


def fit_regime_model(
    features: pd.DataFrame,
    method: str = "K-Means",
    n_regimes: int = 5,
    random_state: int = 42,
) -> RegimeResult:
    """Detect market regimes using clustering and add readable labels."""
    model_matrix = get_model_matrix(features, MODEL_FEATURES)
    feature_cols = model_matrix.columns.tolist()
    aligned = features.loc[model_matrix.index].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(model_matrix)

    if method == "Gaussian Mixture":
        model = GaussianMixture(n_components=n_regimes, covariance_type="full", random_state=random_state)
        clusters = model.fit_predict(X_scaled)
    else:
        model = KMeans(n_clusters=n_regimes, n_init=20, random_state=random_state)
        clusters = model.fit_predict(X_scaled)

    aligned["cluster"] = clusters.astype(int)

    stats_cols = [
        "ret_20d",
        "ret_60d",
        "vol_20d",
        "vol_60d",
        "price_ma_200_gap",
        "drawdown",
        "excess_ret_20d",
    ]
    stats_cols = [col for col in stats_cols if col in aligned.columns]
    cluster_stats = aligned.groupby("cluster")[stats_cols].mean()

    # Ensure columns needed for mapping exist even if a dataset has no benchmark.
    for needed_col in ["ret_60d", "vol_20d", "price_ma_200_gap", "excess_ret_20d"]:
        if needed_col not in cluster_stats.columns:
            cluster_stats[needed_col] = 0.0

    cluster_to_regime = _map_clusters_to_regimes(cluster_stats)
    aligned["regime"] = aligned["cluster"].map(cluster_to_regime)
    aligned["regime_color"] = aligned["regime"].map(REGIME_COLORS)

    return RegimeResult(
        data=aligned,
        model=model,
        scaler=scaler,
        feature_cols=feature_cols,
        cluster_to_regime=cluster_to_regime,
        cluster_stats=cluster_stats,
    )


def current_regime(regime_data: pd.DataFrame) -> dict[str, object]:
    latest = regime_data.dropna(subset=["regime"]).iloc[-1]
    return {
        "date": latest.name.date().isoformat(),
        "price": float(latest["price"]),
        "regime": str(latest["regime"]),
        "volatility": float(latest["vol_20d"]),
        "drawdown": float(latest["drawdown"]),
        "ret_60d": float(latest["ret_60d"]),
    }
