from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.regime_model import REGIME_COLORS


def regime_price_chart(data: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    # Thin black price line for continuity.
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["price"],
            mode="lines",
            line=dict(width=1, color="rgba(80,80,80,0.35)"),
            name=f"{ticker} price",
            hovertemplate="%{x|%Y-%m-%d}<br>Price: $%{y:.2f}<extra></extra>",
        )
    )
    for regime, color in REGIME_COLORS.items():
        part = data[data["regime"] == regime]
        if part.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=part.index,
                y=part["price"],
                mode="markers",
                marker=dict(size=4, color=color),
                name=regime,
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>Price: $%{y:.2f}<br>Regime: "
                    + regime
                    + "<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=f"{ticker} price colored by detected market regime",
        xaxis_title="Date",
        yaxis_title="Adjusted close price",
        legend_title="Regime",
        hovermode="x unified",
        template="plotly_white",
        height=520,
    )
    return fig


def normalized_benchmark_chart(data: pd.DataFrame, ticker: str, benchmark: str) -> go.Figure:
    d = data.dropna(subset=["price", "benchmark_price"]).copy()
    d[f"{ticker} normalized"] = d["price"] / d["price"].iloc[0] * 100
    d[f"{benchmark} normalized"] = d["benchmark_price"] / d["benchmark_price"].iloc[0] * 100
    plot_df = pd.DataFrame({
        "Date": d.index,
        f"{ticker} normalized": d[f"{ticker} normalized"].values,
        f"{benchmark} normalized": d[f"{benchmark} normalized"].values,
    }).melt(id_vars="Date", var_name="Series", value_name="Growth of $100")
    fig = px.line(plot_df, x="Date", y="Growth of $100", color="Series", template="plotly_white")
    fig.update_layout(
        title=f"Growth of $100: {ticker} vs {benchmark}",
        yaxis_title="Value of $100 investment",
        height=420,
    )
    return fig


def volatility_drawdown_chart(data: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["vol_20d"],
            mode="lines",
            name="20-day annualized volatility",
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["drawdown"],
            mode="lines",
            name="Drawdown",
            yaxis="y2",
        )
    )
    fig.update_layout(
        title="Risk view: rolling volatility and drawdown",
        xaxis_title="Date",
        yaxis=dict(title="Volatility", tickformat=".0%"),
        yaxis2=dict(title="Drawdown", tickformat=".0%", overlaying="y", side="right"),
        template="plotly_white",
        hovermode="x unified",
        height=430,
    )
    return fig


def regime_distribution_chart(data: pd.DataFrame) -> go.Figure:
    counts = data["regime"].value_counts().reindex(list(REGIME_COLORS.keys())).dropna()
    fig = px.bar(
        x=counts.index,
        y=counts.values,
        color=counts.index,
        color_discrete_map=REGIME_COLORS,
        labels={"x": "Regime", "y": "Trading days"},
        template="plotly_white",
    )
    fig.update_layout(title="How often each regime appeared", showlegend=False, height=380)
    return fig


def forecast_bar_chart(forecast_df: pd.DataFrame) -> go.Figure:
    if forecast_df.empty:
        return go.Figure()
    fig = px.bar(
        forecast_df,
        x="Confidence",
        y="Horizon",
        color="Predicted Regime",
        orientation="h",
        color_discrete_map=REGIME_COLORS,
        text=forecast_df["Confidence"].map(lambda x: f"{x:.0%}"),
        template="plotly_white",
    )
    fig.update_layout(
        title="Future regime outlook by horizon",
        xaxis=dict(tickformat=".0%", range=[0, 1]),
        height=360,
    )
    return fig


def feature_importance_chart(importance_df: pd.DataFrame) -> go.Figure:
    top = importance_df.head(12).sort_values("importance")
    fig = px.bar(
        top,
        x="importance",
        y="feature",
        orientation="h",
        template="plotly_white",
        labels={"importance": "Average importance", "feature": "Feature"},
    )
    fig.update_layout(title="Most important features for future regime prediction", height=440)
    return fig
