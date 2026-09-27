# MarketMind AI

**Stock Market Regime Detection Dashboard using Machine Learning**

MarketMind AI is a no-API Streamlit project that analyzes a company's historical stock behavior and classifies market conditions into readable regimes:

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

---

## Why this version has no API

Instead of building a FastAPI backend and a separate frontend, this project uses **Streamlit only**.

That means:

- easier setup
- fewer files to debug
- one command to run the app
- dashboard + ML model inside one project
- perfect for a course project, GitHub portfolio, and demo video

You can always add FastAPI later after the ML pipeline and dashboard are stable.

---

## Final Project Idea

**MarketMind AI: Stock Market Regime Detection and Future Trend Classification**

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

AAPL is a good project stock because it has long historical data, clear growth periods, major drawdowns, high public interest, and enough volatility to make regime detection interesting.

The app now supports normal company-name search. Users can select **Apple**, **Nvidia**, **Tesla**, **Microsoft**, **Amazon**, and other common companies from a searchable dropdown. The app converts the company name into the ticker internally because financial data providers require ticker symbols.

---

## Project Structure

```text
MarketMindAI/
│
├── streamlit_app.py
├── requirements.txt
├── README.md
├── Dockerfile
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── ticker_lookup.py
│   ├── features.py
│   ├── regime_model.py
│   ├── forecasting.py
│   ├── plots.py
│   └── summaries.py
│
├── reports/
│   └── figures/
│
└── notebooks/
    └── project_plan.md
```

---

## What the App Does

### 1. Downloads market data

Uses `yfinance` to download:

- selected company stock data
- benchmark data such as SPY, QQQ, DIA, or IWM
- VIX volatility index data

Normal users select a company by name. Internally, the app converts the selected company into a ticker such as Apple → AAPL or Nvidia → NVDA.

### 2. Creates ML features

The feature pipeline creates:

- 1-day, 5-day, 20-day, 60-day, and 120-day returns
- rolling volatility
- moving-average gaps
- price momentum
- drawdown
- volume z-score
- benchmark return
- excess return vs benchmark
- VIX features

### 3. Detects hidden market regimes

The app supports:

- K-Means clustering
- Gaussian Mixture Model clustering

The hidden clusters are mapped to readable labels:

- Bullish
- Bearish
- Sideways
- High Volatility
- Low Volatility

### 4. Forecasts future regime

The app trains Random Forest classifiers to predict the future regime at:

- 5 trading days
- 20 trading days
- 60 trading days
- 252 trading days

### 5. Generates charts and summaries

The dashboard shows:

- price chart colored by market regime
- AAPL vs SPY normalized comparison
- volatility and drawdown chart
- regime distribution
- future regime confidence chart
- feature importance chart
- human-readable text explanation

---

## How to Run Locally

### Step 1: Create a virtual environment

```bash
python -m venv .venv
```

### Step 2: Activate it

Mac/Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the dashboard

```bash
streamlit run streamlit_app.py
```

Then open the local Streamlit URL shown in the terminal.

---

## How to Deploy Without an API

### Option 1: Streamlit Community Cloud

Best beginner option.

1. Push this project to GitHub.
2. Go to Streamlit Community Cloud.
3. Select your GitHub repo.
4. Set main file as:

```text
streamlit_app.py
```

5. Deploy.

### Option 2: Docker

Use Docker if you want a more production-style project.

```bash
docker build -t marketmind-ai .
docker run -p 8501:8501 marketmind-ai
```

Then open:

```text
http://localhost:8501
```

---

## Good YouTube / Tutorial Links to Learn From

Use these to understand how similar projects are built:

1. Streamlit ML app tutorial  
   https://www.youtube.com/watch?v=LJ6DcLGQ4vY

2. Streamlit Python ML course  
   https://www.youtube.com/watch?v=NfwfiyMi1lk

3. Stock prediction Streamlit app tutorial  
   https://www.python-engineer.com/posts/stockprediction-app/

4. Financial dashboard with Streamlit and yfinance  
   https://softhints.com/build-a-financial-dashboard-in-streamlit-step-by-step-guide-with-yfinance/

5. Market regime detection with HMM and Random Forest  
   https://blog.quantinsti.com/regime-adaptive-trading-python/

6. HMM market regime detection project example  
   https://isaacnicas.github.io/market-regime-detection-hmm/

7. Python market regime HMM guide  
   https://www.pythonandtrading.com/detect-market-regimes-hmm-python/

Recommended learning order:

1. Watch Streamlit basics.
2. Watch stock dashboard examples.
3. Read market regime detection examples.
4. Then study this project file by file.

---

## How to Explain This Project in Class

This project does not try to predict the exact stock price. Instead, it detects the market condition or “regime.” This is more realistic because stock prices are noisy, but broader market behavior such as bullish, bearish, sideways, or high-volatility phases can be detected using patterns in returns, volatility, momentum, drawdown, volume, benchmark movement, and volatility index data.

The project first uses unsupervised learning to discover hidden clusters in the historical data. Then those clusters are mapped into readable finance labels. After that, a supervised Random Forest model is trained to predict what regime may appear in future time horizons.

---

## Resume Bullets

**MarketMind AI: Stock Market Regime Detection Dashboard**

- Built a Streamlit-based ML dashboard to classify stock market behavior into Bullish, Bearish, Sideways, High Volatility, and Low Volatility regimes using historical OHLCV data.
- Engineered time-series features including rolling returns, volatility, moving-average gaps, momentum, drawdown, volume z-scores, benchmark returns, and VIX-based indicators.
- Implemented K-Means and Gaussian Mixture clustering for unsupervised regime detection, then trained Random Forest models to forecast future regimes across 1-week, 1-month, 3-month, and 1-year horizons.
- Visualized 30-year stock trends, benchmark comparison, drawdowns, volatility, regime distribution, future confidence, feature importance, and natural-language summaries for non-technical users.

---

## Disclaimer

This project is for educational and research purposes only. It does not provide financial advice, investment recommendations, or guaranteed trading results. Financial markets are uncertain and model outputs should be interpreted as analytical support only.
