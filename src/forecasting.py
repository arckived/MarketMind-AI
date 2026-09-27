from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit


HORIZON_LABELS = {
    5: "Next 1 week",
    20: "Next 1 month",
    60: "Next 3 months",
    252: "Next 1 year",
}


@dataclass
class HorizonModelResult:
    horizon: int
    label: str
    model: RandomForestClassifier
    predicted_regime: str
    confidence: float
    probabilities: dict[str, float]
    accuracy: float
    balanced_accuracy: float
    f1_macro: float
    train_rows: int
    test_rows: int


def train_horizon_forecasters(
    regime_data: pd.DataFrame,
    feature_cols: list[str],
    horizons: list[int] | None = None,
    random_state: int = 42,
) -> list[HorizonModelResult]:
    """Train horizon-specific classifiers to predict future regime labels.

    For each horizon, the target is the regime label shifted backward by that many
    trading days. This asks: given today's features, what regime appears later?
    """
    horizons = horizons or [5, 20, 60, 252]
    results: list[HorizonModelResult] = []

    X_base = regime_data[feature_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill()
    current_X = X_base.iloc[[-1]]

    for horizon in horizons:
        y = regime_data["regime"].shift(-horizon)
        valid = X_base.notna().all(axis=1) & y.notna()
        X = X_base.loc[valid]
        y_valid = y.loc[valid]

        if len(X) < 600 or y_valid.nunique() < 2:
            continue

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y_valid.iloc[:split_idx], y_valid.iloc[split_idx:]

        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=12,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        proba_values = model.predict_proba(current_X)[0]
        classes = model.classes_.tolist()
        probabilities = {str(cls): float(prob) for cls, prob in zip(classes, proba_values)}
        predicted_regime = max(probabilities, key=probabilities.get)
        confidence = probabilities[predicted_regime]

        results.append(
            HorizonModelResult(
                horizon=horizon,
                label=HORIZON_LABELS.get(horizon, f"Next {horizon} trading days"),
                model=model,
                predicted_regime=predicted_regime,
                confidence=float(confidence),
                probabilities=probabilities,
                accuracy=float(accuracy_score(y_test, pred)),
                balanced_accuracy=float(balanced_accuracy_score(y_test, pred)),
                f1_macro=float(f1_score(y_test, pred, average="macro", zero_division=0)),
                train_rows=len(X_train),
                test_rows=len(X_test),
            )
        )

    return results


def forecast_results_to_frame(results: list[HorizonModelResult]) -> pd.DataFrame:
    rows = []
    for item in results:
        rows.append(
            {
                "Horizon": item.label,
                "Trading Days": item.horizon,
                "Predicted Regime": item.predicted_regime,
                "Confidence": item.confidence,
                "Test Accuracy": item.accuracy,
                "Balanced Accuracy": item.balanced_accuracy,
                "F1 Macro": item.f1_macro,
            }
        )
    return pd.DataFrame(rows)


def average_feature_importance(results: list[HorizonModelResult], feature_cols: list[str]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame(columns=["feature", "importance"])
    importances = np.zeros(len(feature_cols), dtype=float)
    used = 0
    for result in results:
        if hasattr(result.model, "feature_importances_"):
            importances += result.model.feature_importances_
            used += 1
    if used == 0:
        return pd.DataFrame(columns=["feature", "importance"])
    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": importances / used,
        }
    ).sort_values("importance", ascending=False)
    return importance_df
