from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class CompanyInfo:
    name: str
    ticker: str
    sector: str = ""


# Common US-listed companies for demo and beginner use.
# Users can still choose Custom and type any valid Yahoo Finance ticker.
COMPANIES: List[CompanyInfo] = [
    CompanyInfo("Apple", "AAPL", "Technology"),
    CompanyInfo("Microsoft", "MSFT", "Technology"),
    CompanyInfo("Nvidia", "NVDA", "Semiconductors / AI"),
    CompanyInfo("Amazon", "AMZN", "E-commerce / Cloud"),
    CompanyInfo("Alphabet / Google", "GOOGL", "Internet / AI"),
    CompanyInfo("Meta Platforms", "META", "Social / AI"),
    CompanyInfo("Tesla", "TSLA", "EV / Energy"),
    CompanyInfo("Netflix", "NFLX", "Streaming"),
    CompanyInfo("JPMorgan Chase", "JPM", "Banking"),
    CompanyInfo("Bank of America", "BAC", "Banking"),
    CompanyInfo("Visa", "V", "Payments"),
    CompanyInfo("Mastercard", "MA", "Payments"),
    CompanyInfo("Coca-Cola", "KO", "Consumer Defensive"),
    CompanyInfo("PepsiCo", "PEP", "Consumer Defensive"),
    CompanyInfo("Walmart", "WMT", "Retail"),
    CompanyInfo("Costco", "COST", "Retail"),
    CompanyInfo("Disney", "DIS", "Entertainment"),
    CompanyInfo("Nike", "NKE", "Consumer"),
    CompanyInfo("McDonald's", "MCD", "Restaurants"),
    CompanyInfo("Starbucks", "SBUX", "Restaurants"),
    CompanyInfo("Berkshire Hathaway", "BRK-B", "Holding Company"),
    CompanyInfo("Johnson & Johnson", "JNJ", "Healthcare"),
    CompanyInfo("UnitedHealth", "UNH", "Healthcare"),
    CompanyInfo("Pfizer", "PFE", "Healthcare"),
    CompanyInfo("Exxon Mobil", "XOM", "Energy"),
    CompanyInfo("Chevron", "CVX", "Energy"),
    CompanyInfo("Ford", "F", "Automotive"),
    CompanyInfo("General Motors", "GM", "Automotive"),
    CompanyInfo("Intel", "INTC", "Semiconductors"),
    CompanyInfo("AMD", "AMD", "Semiconductors"),
    CompanyInfo("Salesforce", "CRM", "Software"),
    CompanyInfo("Oracle", "ORCL", "Software"),
    CompanyInfo("Adobe", "ADBE", "Software"),
]

BENCHMARKS: Dict[str, str] = {
    "S&P 500 ETF / Broad US market (SPY)": "SPY",
    "Nasdaq 100 ETF / Tech-heavy market (QQQ)": "QQQ",
    "Dow Jones ETF / Large industrials (DIA)": "DIA",
    "Russell 2000 ETF / Small caps (IWM)": "IWM",
}


def company_labels() -> List[str]:
    return [f"{c.name} ({c.ticker}) — {c.sector}" for c in COMPANIES] + ["Custom company / ticker"]


def label_to_ticker(label: str) -> str:
    if label == "Custom company / ticker":
        return ""
    start = label.find("(")
    end = label.find(")", start)
    if start == -1 or end == -1:
        return label.upper().strip()
    return label[start + 1 : end].upper().strip()


def ticker_to_company_name(ticker: str) -> str:
    ticker = ticker.upper().strip()
    for company in COMPANIES:
        if company.ticker == ticker:
            return company.name
    return ticker
