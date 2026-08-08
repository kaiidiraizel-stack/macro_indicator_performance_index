"""
data_fetcher.py
----------------
Live data acquisition for the Macro Performance Indicator & Predictor.

Sources (all free, no API key required):
  - World Bank API   -> GDP growth, Inflation (CPI), Forex reserves  (annual)
  - yfinance         -> FX rate + currency volatility, Gold price    (daily)
  - FRED (CSV export)-> Policy / discount rate                       (monthly, India & US only)

No single free source publishes all six indicators at high frequency, so
each is pulled from whichever authoritative source actually has it, then
resampled onto a common monthly index (forward-filled between real
releases — this is genuinely how infrequent official stats work, not a
simplification hiding anything).
"""

import numpy as np
import pandas as pd
import requests
import streamlit as st
from io import StringIO

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

# ----------------------------------------------------------------------
# Source configuration
# ----------------------------------------------------------------------
WB_INDICATORS = {
    "gdp_growth": "NY.GDP.MKTP.KD.ZG",   # GDP growth (annual %)
    "inflation": "FP.CPI.TOTL.ZG",       # Inflation, consumer prices (annual %)
    "forex_reserves": "FI.RES.TOTL.CD",  # Total reserves incl. gold (current US$)
}

# FRED series confirmed to exist for these countries (IMF/IFS discount rate,
# Fed funds rate). Extend cautiously — wrong series IDs fail silently by
# returning no data, but it's still worth keeping this list conservative.
FRED_POLICY_RATE = {
    "IN": "INTDSRINM193N",  # Interest Rate, Discount Rate for India (IMF/IFS, monthly)
    "US": "FEDFUNDS",       # Federal Funds Effective Rate (monthly)
}

FX_TICKER = {
    "IN": "INR=X",
    "US": "DX-Y.NYB",  # no USD/USD pair exists — use the Dollar Index instead
    "GB": "GBP=X",
    "CN": "CNY=X",
    "JP": "JPY=X",
    "BR": "BRL=X",
}

COUNTRIES = {
    "India": "IN",
    "United States": "US",
    "United Kingdom": "GB",
    "China": "CN",
    "Japan": "JP",
    "Brazil": "BR",
}

INDICATOR_UPDATE_FREQ = {
    "gdp_growth": "annual (World Bank)",
    "inflation": "annual (World Bank)",
    "forex_reserves": "annual (World Bank)",
    "fx_rate": "daily (Yahoo Finance)",
    "currency_volatility": "daily, 21-day rolling (Yahoo Finance)",
    "gold_price": "daily (Yahoo Finance, USD/oz)",
    "repo_rate": "monthly, India/US only (FRED)",
}


# ----------------------------------------------------------------------
# Individual fetchers — each cached, each fails soft (returns empty Series)
# ----------------------------------------------------------------------
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_worldbank_series(country_code: str, indicator_code: str, per_page: int = 100) -> pd.Series:
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}"
    try:
        r = requests.get(url, params={"format": "json", "per_page": per_page}, timeout=15)
        r.raise_for_status()
        data = r.json()
        if len(data) < 2 or not data[1]:
            return pd.Series(dtype=float)
        records = {item["date"]: item["value"] for item in data[1] if item["value"] is not None}
        if not records:
            return pd.Series(dtype=float)
        s = pd.Series(records)
        s.index = pd.to_datetime(s.index, format="%Y")
        return s.sort_index()
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yf_history(ticker: str, period: str = "10y") -> pd.Series:
    if yf is None:
        return pd.Series(dtype=float)
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
        if df is None or df.empty:
            return pd.Series(dtype=float)
        close = df["Close"]
        if isinstance(close, pd.DataFrame):  # yfinance sometimes returns multi-col
            close = close.iloc[:, 0]
        return close.dropna()
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_fred_series(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"])
        s = pd.to_numeric(df.set_index("date")["value"], errors="coerce").dropna()
        return s
    except Exception:
        return pd.Series(dtype=float)


# ----------------------------------------------------------------------
# Combine everything into one monthly-indexed DataFrame
# ----------------------------------------------------------------------
def _resample_to_monthly(series: pd.Series, monthly_index: pd.DatetimeIndex, how: str = "ffill_annual") -> pd.Series:
    if series.empty:
        return pd.Series(index=monthly_index, dtype=float)
    if how == "ffill_annual":
        union_idx = monthly_index.union(series.index)
        return series.reindex(union_idx).sort_index().ffill().reindex(monthly_index)
    elif how == "last_of_month":
        resampled = series.resample("MS").last()
        return resampled.reindex(monthly_index).ffill()
    return pd.Series(index=monthly_index, dtype=float)


def build_live_dataset(country_name: str = "India", years: int = 10):
    """Fetches all available live indicators for a country and returns
    (dataframe, notes) where notes lists which indicators came through
    and which didn't."""
    code = COUNTRIES.get(country_name, "IN")
    end = pd.Timestamp.today().normalize().replace(day=1)
    start = end - pd.DateOffset(years=years)
    monthly_index = pd.date_range(start, end, freq="MS")

    df = pd.DataFrame(index=monthly_index)
    notes = {}

    # --- World Bank: GDP growth, inflation, forex reserves (annual) ---
    for col, wb_code in WB_INDICATORS.items():
        s = fetch_worldbank_series(code, wb_code)
        if s.empty:
            notes[col] = "unavailable"
            continue
        monthly = _resample_to_monthly(s, monthly_index, "ffill_annual")
        if col == "forex_reserves":
            monthly = monthly / 1e9  # USD -> USD Bn
        df[col] = monthly.values
        notes[col] = f"live, last release {s.index[-1].year}"

    # --- yfinance: FX rate + volatility, gold price (daily) ---
    fx_ticker = FX_TICKER.get(code, "INR=X")
    fx = fetch_yf_history(fx_ticker)
    if not fx.empty:
        df["fx_rate"] = _resample_to_monthly(fx, monthly_index, "last_of_month").values
        daily_ret = fx.pct_change().dropna()
        ann_vol = (daily_ret.rolling(21).std() * np.sqrt(252) * 100).dropna()
        df["currency_volatility"] = _resample_to_monthly(ann_vol, monthly_index, "last_of_month").values
        notes["fx_rate"] = notes["currency_volatility"] = "live (Yahoo Finance)"
    else:
        notes["fx_rate"] = notes["currency_volatility"] = "unavailable"

    gold = fetch_yf_history("GC=F")
    if not gold.empty:
        df["gold_price"] = _resample_to_monthly(gold, monthly_index, "last_of_month").values
        notes["gold_price"] = "live (Yahoo Finance)"
    else:
        notes["gold_price"] = "unavailable"

    # --- FRED: policy / repo-proxy rate (India & US only) ---
    fred_id = FRED_POLICY_RATE.get(code)
    if fred_id:
        policy = fetch_fred_series(fred_id)
        if not policy.empty:
            df["repo_rate"] = _resample_to_monthly(policy, monthly_index, "ffill_annual").values
            notes["repo_rate"] = f"live, last release {policy.index[-1].strftime('%b %Y')}"
        else:
            notes["repo_rate"] = "unavailable"
    else:
        notes["repo_rate"] = f"no free live source for {country_name}"

    df = df.dropna(axis=1, how="all")
    return df, notes
