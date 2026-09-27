# MarketMind AI Project Plan

## Goal
Build a no-API Streamlit dashboard that detects market regimes for a selected stock and forecasts likely future regimes.

## Main Pipeline

1. Download stock, benchmark, and VIX data.
2. Create technical/time-series features.
3. Use clustering to detect hidden market states.
4. Map hidden states to readable regimes.
5. Train horizon-specific Random Forest classifiers.
6. Display charts and summaries in Streamlit.

## Suggested Final Demo

- Ticker: AAPL
- Benchmark: SPY
- Volatility index: ^VIX
- Start date: 1995-01-01
- Model: K-Means
- Clusters: 5

## Possible Extensions

- Add HMM with hmmlearn.
- Add SHAP explainability.
- Add news sentiment using FinBERT.
- Add backtesting for regime-based strategy.
- Add MLflow experiment tracking.
