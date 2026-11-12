"""Download adjusted daily closes for the study universe into data/prices.csv.

    python scripts/fetch_prices.py
"""
from pathlib import Path

import yfinance as yf

UNIVERSE = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
    "JNJ": "Health Care", "PFE": "Health Care",
    "XOM": "Energy", "CVX": "Energy",
    "PG": "Consumer Staples", "KO": "Consumer Staples",
    "SPY": "Benchmark",
}

LARGE_CAPS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "BRK-B", "JPM", "JNJ", "V", "PG", "XOM", "UNH", "HD",
              "MA", "CVX", "LLY", "PFE", "KO", "PEP", "MRK", "ABBV", "BAC", "WMT", "COST", "DIS", "CSCO", "ORCL",
              "INTC", "ADBE", "CRM", "NFLX", "T", "VZ", "MCD", "NKE", "IBM", "CAT", "GE", "HON"]

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data"
    out.mkdir(exist_ok=True)
    px = yf.download(list(UNIVERSE), start="2012-01-01", auto_adjust=True, progress=False)["Close"]
    px.dropna(how="all").round(4).to_csv(out / "prices.csv")
    print(px.shape, "->", out / "prices.csv")
    big = yf.download(LARGE_CAPS + ["SPY"], start="2013-01-01", end="2026-11-01", auto_adjust=True, progress=False)["Close"]
    big.dropna(how="all").round(4).to_csv(out / "large_caps_2013.csv")
