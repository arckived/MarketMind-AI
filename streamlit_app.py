from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data_loader import load_market_data
from src.ticker_lookup import BENCHMARKS, company_labels, label_to_ticker, ticker_to_company_name
from src.features import build_feature_table
from src.forecasting import (
    average_feature_importance,
    forecast_results_to_frame,
    train_horizon_forecasters,
)
from src.plots import (
    feature_importance_chart,
    forecast_bar_chart,
    normalized_benchmark_chart,
    regime_distribution_chart,
    regime_price_chart,
    volatility_drawdown_chart,
)
from src.regime_model import REGIME_COLORS, current_regime, fit_regime_model
from src.summaries import (
    current_regime_summary,
    forecast_summary,
    historical_summary,
    model_quality_summary,
)

st.set_page_config(
    page_title="MarketMind AI",
    page_icon="📈",
    layout="wide",
)


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_and_prepare(ticker: str, benchmark: str, vix: str, start: str, method: str, n_regimes: int):
    bundle = load_market_data(ticker=ticker, benchmark=benchmark, vix_symbol=vix, start=start)
    features = build_feature_table(bundle.stock, bundle.benchmark_df, bundle.vix)
    regime_result = fit_regime_model(features, method=method, n_regimes=n_regimes)
    forecast_results = train_horizon_forecasters(regime_result.data, regime_result.feature_cols)
    forecast_df = forecast_results_to_frame(forecast_results)
    importance_df = average_feature_importance(forecast_results, regime_result.feature_cols)
    return bundle, regime_result, forecast_results, forecast_df, importance_df


def pct_text(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    st.title("MarketMind AI: Stock Market Regime Detection")
    st.caption(
        "A no-API Streamlit project that detects Bullish, Bearish, Sideways, High Volatility, and Low Volatility market regimes."
    )

    with st.sidebar:
        st.header("Project controls")
        company_choice = st.selectbox(
            "Company name",
            company_labels(),
            index=0,
            help="Start typing a company name like Apple, Nvidia, Tesla, Microsoft, or Amazon.",
        )
        resolved_ticker = label_to_ticker(company_choice)
        if company_choice == "Custom company / ticker":
            ticker = st.text_input("Enter ticker manually", value="AAPL", help="Example: AAPL, MSFT, NVDA, TSLA").upper().strip()
            company_name = ticker_to_company_name(ticker)
        else:
            ticker = resolved_ticker
            company_name = ticker_to_company_name(ticker)
            st.caption(f"Using ticker: **{ticker}**")

        benchmark_choice = st.selectbox("Benchmark", list(BENCHMARKS.keys()), index=0)
        benchmark = BENCHMARKS[benchmark_choice]
        vix_symbol = st.text_input("Volatility index", value="^VIX").strip()
        start = st.text_input("Start date", value="1995-01-01")
        method = st.selectbox("Regime detection model", ["K-Means", "Gaussian Mixture"], index=0)
        n_regimes = st.slider("Number of hidden clusters", min_value=3, max_value=7, value=5)
        st.markdown("---")
        st.write("**Tip:** Search by company name. The app converts it to the market ticker internally.")
        run_button = st.button("Run analysis", type="primary", use_container_width=True)

    if not ticker:
        st.warning("Enter a stock ticker to start.")
        return

    if not run_button:
        st.info("Choose a ticker and click **Run analysis**. The default AAPL setup is ready to go.")
        return

    try:
        with st.spinner("Downloading data, detecting regimes, and training future regime models..."):
            bundle, regime_result, forecast_results, forecast_df, importance_df = load_and_prepare(
                ticker=ticker,
                benchmark=benchmark,
                vix=vix_symbol,
                start=start,
                method=method,
                n_regimes=n_regimes,
            )
    except Exception as exc:
        st.error("The analysis could not be completed. Check the ticker/date and try again.")
        st.exception(exc)
        return

    data = regime_result.data
    current = current_regime(data)
    first_price = data["price"].iloc[0]
    last_price = data["price"].iloc[-1]
    total_return = last_price / first_price - 1
    max_drawdown = data["drawdown"].min()
    latest_vol = data["vol_20d"].iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current regime", current["regime"])
    c2.metric("Latest price", f"${current['price']:.2f}")
    c3.metric("Total return", pct_text(total_return))
    c4.metric("Max drawdown", pct_text(max_drawdown))

    st.markdown(
        f"<div style='padding:12px;border-left:6px solid {REGIME_COLORS.get(current['regime'], '#999')};background:#f8f9fa'>"
        f"{current_regime_summary(company_name, current)}"
        "</div>",
        unsafe_allow_html=True,
    )

    tab_overview, tab_regimes, tab_forecast, tab_model, tab_about = st.tabs(
        ["Overview", "Regime Detection", "Future Outlook", "Model Quality", "How it works"]
    )

    with tab_overview:
        left, right = st.columns([1.35, 1])
        with left:
            st.plotly_chart(regime_price_chart(data, company_name), use_container_width=True)
        with right:
            st.subheader("Text summary")
            st.write(historical_summary(company_name, benchmark, data))
            st.subheader("Regime legend")
            for name, color in REGIME_COLORS.items():
                st.markdown(
                    f"<span style='display:inline-block;width:14px;height:14px;background:{color};border-radius:50%;margin-right:8px'></span>{name}",
                    unsafe_allow_html=True,
                )
        st.plotly_chart(normalized_benchmark_chart(data, company_name, benchmark), use_container_width=True)

    with tab_regimes:
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(regime_distribution_chart(data), use_container_width=True)
        with col_b:
            st.plotly_chart(volatility_drawdown_chart(data), use_container_width=True)

        st.subheader("Cluster-to-regime mapping")
        mapping_df = regime_result.cluster_stats.copy()
        mapping_df["Readable Regime"] = mapping_df.index.map(regime_result.cluster_to_regime)
        st.dataframe(mapping_df, use_container_width=True)

    with tab_forecast:
        st.subheader("Future regime forecast")
        if forecast_df.empty:
            st.warning("Not enough data to train future horizon models.")
        else:
            fc_left, fc_right = st.columns([1, 1])
            with fc_left:
                st.plotly_chart(forecast_bar_chart(forecast_df), use_container_width=True)
            with fc_right:
                st.write(forecast_summary(forecast_results))
                display_df = forecast_df.copy()
                for col in ["Confidence", "Test Accuracy", "Balanced Accuracy", "F1 Macro"]:
                    display_df[col] = display_df[col].map(lambda x: f"{x:.2%}" if col != "F1 Macro" else f"{x:.3f}")
                st.dataframe(display_df, use_container_width=True, hide_index=True)

    with tab_model:
        st.subheader("Model evaluation")
        st.write(model_quality_summary(forecast_results))
        if not forecast_df.empty:
            eval_df = forecast_df[["Horizon", "Test Accuracy", "Balanced Accuracy", "F1 Macro"]].copy()
            st.dataframe(eval_df, use_container_width=True, hide_index=True)
        if not importance_df.empty:
            st.plotly_chart(feature_importance_chart(importance_df), use_container_width=True)
        st.info(
            "The project uses time-ordered splits for evaluation. This is important because stock data should train on the past and test on the future."
        )

    with tab_about:
        st.subheader("Project pipeline")
        st.markdown(
            """
            1. Download stock, benchmark, and VIX data with yfinance.  
            2. Engineer return, volatility, momentum, drawdown, moving-average, benchmark, and volume features.  
            3. Detect hidden market states using K-Means or Gaussian Mixture clustering.  
            4. Convert unsupervised clusters into readable regimes.  
            5. Train Random Forest models to predict the future regime at 1-week, 1-month, 3-month, and 1-year horizons.  
            6. Display charts, summaries, confidence, and evaluation metrics in Streamlit.  
            """
        )
        st.warning(
            "Educational project only. This is not financial advice and should not be used as the only basis for investment decisions."
        )


if __name__ == "__main__":
    main()
