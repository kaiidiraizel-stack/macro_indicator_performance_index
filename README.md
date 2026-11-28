# 📊 Macroeconomic Performance Indicator & Predictor
 
**🔗 Live app: [mpi-ddwnp87ycmzdvt2wynqceg.streamlit.app](https://mpi-ddwnp87ycmzdvt2wynqceg.streamlit.app/)**
 
An AI-assisted Streamlit dashboard that **live-fetches** (no CSV upload
needed) key macroeconomic indicators — **GDP growth, inflation, forex
reserves, gold price, currency volatility, FX rate, and policy/repo rate** —
and forecasts them with machine learning and time-series models. Built to
help spot where an economy is heading next, not just where it's been.
 
## What it does
 
- **🏠 Overview & Risk** — snapshot of all live indicators, a heuristic
  "Economic Stability Score" gauge (z-score blend vs. historical norms) with
  a driver breakdown, and trend charts you can scrub with a date-range
  slider. Click "Forecast this" on any metric to jump straight to it.
- **🔮 Forecast** — pick an indicator, a horizon (3–24 months), and a model
  (Random Forest, Linear Trend, Holt-Winters, or ARIMA). Shows the forecast
  with a confidence band, a backtest accuracy score (MAPE), a **what-if
  scenario simulator** (shock the latest reading ±30% and see the forecast
  respond), and a side-by-side comparison of all four models.
- **🔍 Compare & Explore** — overlay any indicators on a normalized (z-score)
  chart, a full correlation heatmap, and a relationship explorer (scatter +
  OLS trendline between any two indicators).
- **📄 Data** — view, filter, and download the full historical + forecast
  dataset.
## Run it locally
 
```bash
pip install -r requirements.txt
streamlit run app.py
```
 
Open the local URL Streamlit prints (usually `http://localhost:8501`). The
sidebar defaults to **🔴 Live Data (auto-fetch)** — pick a country and go. A
"Refresh live data now" button clears the cache and re-pulls.
 
## Deploying your own copy
 
This is already live on Streamlit Community Cloud at the link above. To
deploy your own fork:
 
1. Push `app.py`, `data_fetcher.py`, and `requirements.txt` to a public
   GitHub repo (all in the same top-level folder).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, click **New app**, pick your repo/branch, and set the main file
   path to `app.py`.
3. Deploy — no secrets or API keys are needed since every data source used
   here is free and keyless.
## Where the live data actually comes from
 
No single free source publishes all these indicators at high frequency, so
`data_fetcher.py` pulls each from whichever authoritative source has it:
 
| Indicator | Source | Frequency |
|---|---|---|
| GDP growth, Inflation, Forex reserves | [World Bank API](https://api.worldbank.org) (no key needed) | Annual, forward-filled monthly |
| FX rate & currency volatility | [Yahoo Finance](https://finance.yahoo.com) via `yfinance` | Daily |
| Gold price (USD/oz) | Yahoo Finance (`GC=F` futures) via `yfinance` | Daily |
| Policy / repo-proxy rate | [FRED](https://fred.stlouisfed.org) CSV export | Monthly — **India & US only**; other countries show "no free live source" |
 
Countries supported out of the box: India, United States, United Kingdom,
China, Japan, Brazil (edit `COUNTRIES` / `FX_TICKER` in `data_fetcher.py` to
add more — World Bank and Yahoo Finance cover most countries; FRED's policy
rate is the one indicator that needs a verified series ID per country).
 
The sidebar has a **"📡 Live source status"** expander showing exactly which
indicators came through live and which didn't for the selected country.
 
**Good faith limitation:** the repo rate isn't published as a free, truly
real-time API by any central bank — the FRED series used is IMF-published
and monthly. If you need the RBI's actual current repo rate, cross-check
against [rbi.org.in](https://www.rbi.org.in) directly.
 
## Fallback behavior
 
If a live source is unreachable (API down, ticker delisted, rate-limited,
etc.), that fetch fails silently and the app falls back to realistic
**synthetic** demo data — it will never crash, it'll just tell you via the
sidebar status panel and switch data sources.
 
## How the forecasting works
 
- **Random Forest / Linear Trend** — supervised regression on lag features
  (last 6 periods) plus a time trend; forecasts recursively, feeding each
  prediction back in as a lag for the next step.
- **Holt-Winters** — classical exponential smoothing with trend + (if enough
  history) yearly seasonality.
- **ARIMA(2,1,2)** — standard autoregressive integrated moving average model.
Each model is backtested by holding out the most recent period it would have
had to forecast blind, then scored with MAPE (mean absolute percentage
error) so you can see which model is currently most reliable for a given
indicator. The **what-if simulator** on the Forecast tab lets you shock the
latest known value and re-run the same model to stress-test the outlook.
 
## Project structure
 
```
app.py            # Streamlit UI — tabs, charts, forecasting orchestration
data_fetcher.py   # Live data acquisition (World Bank / yfinance / FRED)
requirements.txt  # Python dependencies
```
 
## Notes / limitations
 
- Live indicators update at genuinely different real-world frequencies —
  GDP/inflation/reserves are annual releases forward-filled monthly; FX and
  gold are daily. This is a real constraint of official data, not a
  simplification.
- Forecasts are illustrative pattern-extrapolations from historical data,
  not a substitute for expert macroeconomic/policy judgment.
- Synthetic demo data (used only as a fallback) is calibrated to look like a
  large emerging-market economy — it is not real published statistics.
 
