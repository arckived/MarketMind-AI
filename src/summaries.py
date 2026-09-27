from __future__ import annotations

import numpy as np
import pandas as pd

from src.forecasting import HorizonModelResult


def _pct(x: float) -> str:
    if pd.isna(x):
        return "N/A"
    return f"{x * 100:.2f}%"


def historical_summary(ticker: str, benchmark: str, data: pd.DataFrame) -> str:
    d = data.dropna(subset=["price"])
    if len(d) < 2:
        return "Not enough data to generate a historical summary."

    first_date = d.index[0].date().isoformat()
    last_date = d.index[-1].date().isoformat()
    first_price = d["price"].iloc[0]
    last_price = d["price"].iloc[-1]
    total_return = last_price / first_price - 1
    years = max((d.index[-1] - d.index[0]).days / 365.25, 1)
    annualized_return = (last_price / first_price) ** (1 / years) - 1
    max_drawdown = d["drawdown"].min()
    latest_vol = d["vol_20d"].iloc[-1]

    annual = d["price"].resample("YE").last().pct_change().dropna()
    best_year = annual.idxmax().year if not annual.empty else "N/A"
    worst_year = annual.idxmin().year if not annual.empty else "N/A"

    return (
        f"From {first_date} to {last_date}, {ticker} moved from ${first_price:.2f} to ${last_price:.2f}, "
        f"which is a total return of {_pct(total_return)} and an annualized return of {_pct(annualized_return)}. "
        f"The largest historical drawdown in this window was {_pct(max_drawdown)}, while the latest 20-day annualized "
        f"volatility is {_pct(latest_vol)}. The best calendar year in this window was {best_year}, and the weakest was {worst_year}. "
        f"The benchmark comparison with {benchmark} helps show whether the stock moved with or against the broader market."
    )


def current_regime_summary(ticker: str, current: dict[str, object]) -> str:
    regime = str(current.get("regime", "Unknown"))
    ret_60d = float(current.get("ret_60d", np.nan))
    vol = float(current.get("volatility", np.nan))
    drawdown = float(current.get("drawdown", np.nan))

    interpretation = {
        "Bullish": "price momentum is positive and the stock is trading in a stronger trend zone",
        "Bearish": "recent returns and trend features are weak compared with other historical periods",
        "Sideways": "the model does not see a strong upward or downward trend",
        "High Volatility": "price movement is unstable and risk is elevated",
        "Low Volatility": "the stock is moving more calmly than usual",
    }.get(regime, "the model detected a mixed market condition")

    return (
        f"The latest detected regime for {ticker} is **{regime}**. This means {interpretation}. "
        f"The recent 60-day return is {_pct(ret_60d)}, current annualized volatility is {_pct(vol)}, "
        f"and the stock is currently {_pct(drawdown)} below its previous peak."
    )


def forecast_summary(results: list[HorizonModelResult]) -> str:
    if not results:
        return "Not enough labeled history was available to create future regime forecasts."

    fragments = []
    for r in results:
        confidence_word = "high" if r.confidence >= 0.70 else "moderate" if r.confidence >= 0.55 else "low"
        fragments.append(
            f"{r.label}: {r.predicted_regime} with {confidence_word} confidence ({r.confidence:.0%})"
        )
    return (
        "Future regime outlook: " + "; ".join(fragments) + ". "
        "These are regime classifications, not guaranteed price targets or financial advice."
    )


def model_quality_summary(results: list[HorizonModelResult]) -> str:
    if not results:
        return "Model quality could not be evaluated because there was not enough training data."
    best = max(results, key=lambda r: r.f1_macro)
    return (
        f"The strongest horizon by macro F1 is {best.label}, with F1={best.f1_macro:.3f}, "
        f"accuracy={best.accuracy:.3f}, and balanced accuracy={best.balanced_accuracy:.3f}. "
        "Balanced accuracy and macro F1 matter here because market regimes are usually imbalanced."
    )
