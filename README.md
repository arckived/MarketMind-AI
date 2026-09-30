# MarketMind AI

**Stock Market Regime Detection Dashboard using Machine Learning**

MarketMind AI is a Streamlit project that analyzes a company's historical stock behavior and classifies market conditions into readable regimes:

- Bullish
- Bearish
- Sideways
- High Volatility
- Low Volatility

It also predicts the likely future regime over multiple horizons:

- Next 1 week
- Next 1 month
- Next 3 months
- Next 1 year

> This project is for learning, research, and portfolio use only. It is not financial advice.


## Ideology 

The system answers:

1. What regime is the stock currently in?
2. How did regimes change over the past 30 years?
3. Was the stock bullish, bearish, sideways, calm, or highly volatile?
4. What future regime is likely in 1 week, 1 month, 3 months, and 1 year?
5. Which features influenced the future-regime prediction?
6. Can normal users understand the result through charts and text summaries?

---

## Recommended Demo Setup

Use this default setup:

```text
Company name: Apple
Internal ticker used by app: AAPL
Benchmark: S&P 500 ETF / Broad US market (SPY)
Volatility index: ^VIX
Start date: 1995-01-01
Model: K-Means
Clusters: 5
```


## Disclaimer

This project is for educational and research purposes only. It does not provide financial advice, investment recommendations, or guaranteed trading results. Financial markets are uncertain and model outputs should be interpreted as analytical support only.
