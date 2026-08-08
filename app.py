"""
Macroeconomic Performance Indicator & Predictor
------------------------------------------------
An AI-assisted dashboard that tracks and forecasts key macroeconomic
indicators — GDP growth, inflation, forex reserves, gold price,
currency volatility, FX rate, and policy/repo rate — using classical
ML / time-series models (Linear Trend, Random Forest, Holt-Winters, ARIMA).

Data is fetched live (World Bank API, Yahoo Finance, FRED) — no CSV
upload required. Falls back to synthetic demo data automatically if a
live source is unreachable.

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import warnings

from data_fetcher import build_live_dataset, COUNTRIES, INDICATOR_UPDATE_FREQ

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Macro Performance Indicator & Predictor",
    page_icon="📊",
    layout="wide",
)

INDICATORS = {
    "GDP Growth (%)": "gdp_growth",
    "Inflation / CPI (%)": "inflation",
    "Forex Reserves (USD Bn)": "forex_reserves",
    "Gold Price (USD/oz)": "gold_price",
    "Currency Volatility (Ann. %)": "currency_volatility",
    "FX Rate (local currency per USD)": "fx_rate",
    "Policy / Repo Rate (%)": "repo_rate",
}

# direction that's "good" for the stability score: +1 higher-is-better,
# -1 lower-is-better, 0 closer-to-its-own-history-is-better
INDICATOR_POLARITY = {
    "gdp_growth": 1,
    "inflation": -1,
    "forex_reserves": 1,
    "gold_price": 1,
    "currency_volatility": -1,
    "fx_rate": 0,
    "repo_rate": 0,
}

# ----------------------------------------------------------------------
# DATA — synthetic-but-realistic macro series, or user upload
# ----------------------------------------------------------------------
@st.cache_data
def generate_synthetic_data(n_years: int = 12, seed: int = 42) -> pd.DataFrame:
    """Generates monthly macro data with trend, cycles, seasonality and noise,
    loosely calibrated to look like a large emerging-market economy."""
    rng = np.random.default_rng(seed)
    periods = n_years * 12
    dates = pd.date_range(end=pd.Timestamp.today().normalize().replace(day=1),
                           periods=periods, freq="MS")
    t = np.arange(periods)

    # business cycle component shared across series (drives correlation)
    cycle = np.sin(2 * np.pi * t / 48) * 1.0 + np.sin(2 * np.pi * t / 96) * 0.5

    gdp_growth = (
        6.0
        + 1.2 * cycle
        + 0.4 * np.sin(2 * np.pi * t / 12)  # seasonality
        + rng.normal(0, 0.35, periods)
    )
    # simulate a couple of shock periods (e.g. pandemic-like dip)
    shock_idx = periods - rng.integers(60, 90)
    gdp_growth[shock_idx:shock_idx + 6] -= np.linspace(0, 8, 6)
    gdp_growth[shock_idx + 6:shock_idx + 18] += np.linspace(0, 4, 12)

    inflation = (
        5.5
        - 0.6 * cycle
        + 0.3 * np.sin(2 * np.pi * t / 12 + 1)
        + rng.normal(0, 0.25, periods)
    )
    inflation = np.clip(inflation, 0.5, None)

    repo_rate = np.zeros(periods)
    repo_rate[0] = 6.0
    for i in range(1, periods):
        # policy responds to inflation with lag + smoothing (Taylor-rule-ish)
        target = 2.0 + 1.3 * inflation[max(0, i - 3)]
        repo_rate[i] = 0.92 * repo_rate[i - 1] + 0.08 * target + rng.normal(0, 0.05)
    repo_rate = np.clip(repo_rate, 3.5, 9.5)

    forex_reserves = (
        400
        + np.cumsum(rng.normal(0.8, 3.0, periods))
        + 15 * cycle
    )
    forex_reserves = np.clip(forex_reserves, 250, None)

    gold_price = (
        1800
        + np.cumsum(np.abs(rng.normal(2.2, 6.0, periods)))  # long-run uptrend
        + 40 * cycle
    )

    fx_rate = 78 + np.cumsum(rng.normal(0.05, 0.35, periods)) + 1.5 * cycle
    fx_rate = np.clip(fx_rate, 60, None)

    currency_volatility = (
        12
        + 3 * np.abs(cycle)
        + 4 * np.abs(rng.normal(0, 1, periods))
        + (gdp_growth < 2) * 6  # spikes when growth is weak
    )
    currency_volatility = np.clip(currency_volatility, 4, None)

    df = pd.DataFrame(
        {
            "date": dates,
            "gdp_growth": gdp_growth,
            "inflation": inflation,
            "forex_reserves": forex_reserves,
            "gold_price": gold_price,
            "currency_volatility": currency_volatility,
            "fx_rate": fx_rate,
            "repo_rate": repo_rate,
        }
    ).set_index("date")
    return df.round(2)


# ----------------------------------------------------------------------
# FORECASTING MODELS
# ----------------------------------------------------------------------
def make_lag_features(series: pd.Series, n_lags: int = 6) -> pd.DataFrame:
    df = pd.DataFrame({"y": series})
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df["t"] = np.arange(len(df))
    return df.dropna()


def forecast_linear_trend(series: pd.Series, horizon: int):
    n_lags = 6
    feat = make_lag_features(series, n_lags)
    X = feat.drop(columns="y")
    y = feat["y"]
    model = LinearRegression().fit(X, y)

    history = list(series.values[-n_lags:])
    t_next = feat["t"].iloc[-1] + 1
    preds = []
    for _ in range(horizon):
        row = {f"lag_{lag}": history[-lag] for lag in range(1, n_lags + 1)}
        row["t"] = t_next
        x = pd.DataFrame([row])[X.columns]
        p = model.predict(x)[0]
        preds.append(p)
        history.append(p)
        t_next += 1
    return np.array(preds), "Linear Trend (lag regression)"


def forecast_random_forest(series: pd.Series, horizon: int):
    n_lags = 6
    feat = make_lag_features(series, n_lags)
    X = feat.drop(columns="y")
    y = feat["y"]
    model = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)
    model.fit(X, y)

    history = list(series.values[-n_lags:])
    t_next = feat["t"].iloc[-1] + 1
    preds = []
    for _ in range(horizon):
        row = {f"lag_{lag}": history[-lag] for lag in range(1, n_lags + 1)}
        row["t"] = t_next
        x = pd.DataFrame([row])[X.columns]
        p = model.predict(x)[0]
        preds.append(p)
        history.append(p)
        t_next += 1
    return np.array(preds), "Random Forest (ML pattern detection)"


def forecast_holt_winters(series: pd.Series, horizon: int):
    seasonal_periods = 12 if len(series) >= 24 else None
    model = ExponentialSmoothing(
        series,
        trend="add",
        seasonal="add" if seasonal_periods else None,
        seasonal_periods=seasonal_periods,
    ).fit()
    preds = model.forecast(horizon)
    return preds.values, "Holt-Winters Exponential Smoothing"


def forecast_arima(series: pd.Series, horizon: int):
    model = ARIMA(series, order=(2, 1, 2)).fit()
    preds = model.forecast(horizon)
    return preds.values, "ARIMA(2,1,2)"


MODEL_FUNCS = {
    "Random Forest": forecast_random_forest,
    "Linear Trend": forecast_linear_trend,
    "Holt-Winters": forecast_holt_winters,
    "ARIMA": forecast_arima,
}


def historical_backtest_error(series: pd.Series, model_func, horizon: int) -> float:
    """Quick backtest: hold out last `horizon` points, forecast, return MAPE."""
    if len(series) < horizon + 24:
        return np.nan
    train = series.iloc[: -horizon]
    test = series.iloc[-horizon:]
    try:
        preds, _ = model_func(train, horizon)
        mape = np.mean(np.abs((test.values - preds) / np.where(test.values == 0, 1, test.values))) * 100
        return round(mape, 2)
    except Exception:
        return np.nan


# ----------------------------------------------------------------------
# RISK SCORING
# ----------------------------------------------------------------------
def compute_risk_score(df: pd.DataFrame) -> tuple:
    """Blends whichever indicator columns are actually present into a
    heuristic 0-100 stability score. Robust to missing columns, since
    live-fetched data won't always have every indicator available."""
    latest = df.iloc[-1]
    hist = df.iloc[:-1] if len(df) > 1 else df

    components = {}
    for col, polarity in INDICATOR_POLARITY.items():
        if col not in df.columns:
            continue
        series = hist[col].dropna()
        if series.empty or pd.isna(latest[col]):
            continue
        mu, sigma = series.mean(), series.std()
        if not sigma or np.isnan(sigma):
            continue
        val = (latest[col] - mu) / sigma
        label = {v: k for k, v in INDICATORS.items()}.get(col, col)
        components[label] = -abs(val) if polarity == 0 else polarity * val

    if not components:
        return 50.0, {}

    raw_score = np.mean(list(components.values()))
    stability_score = float(np.clip(50 + raw_score * 15, 0, 100))
    return stability_score, components


# ----------------------------------------------------------------------
# SIDEBAR — controls
# ----------------------------------------------------------------------
st.sidebar.title("⚙️ Controls")

data_source = st.sidebar.radio(
    "Data source", ["🔴 Live Data (auto-fetch)", "Synthetic demo data"], index=0
)

fetch_notes = {}
if data_source == "🔴 Live Data (auto-fetch)":
    country_name = st.sidebar.selectbox("Country", list(COUNTRIES.keys()), index=0)
    live_years = st.sidebar.slider("Years of history", 3, 20, 10)
    if st.sidebar.button("🔄 Refresh live data now"):
        st.cache_data.clear()
    with st.spinner(f"Fetching live data for {country_name}..."):
        data, fetch_notes = build_live_dataset(country_name, years=live_years)
    if data.empty or data.shape[1] == 0:
        st.sidebar.error("Live fetch failed for every indicator — falling back to synthetic data.")
        data = generate_synthetic_data(n_years=live_years)
        data_source = "Synthetic demo data"
else:
    n_years = st.sidebar.slider("Years of history (synthetic)", 5, 20, 12)
    data = generate_synthetic_data(n_years=n_years)

available_cols = [c for c in INDICATORS.values() if c in data.columns]
label_lookup = {v: k for k, v in INDICATORS.items()}

if not available_cols:
    st.error("No indicator data available at all. Try Synthetic demo data from the sidebar.")
    st.stop()

st.sidebar.markdown("---")
min_date, max_date = data.index.min().to_pydatetime(), data.index.max().to_pydatetime()
view_start, view_end = st.sidebar.slider(
    "🕘 Chart date range",
    min_value=min_date, max_value=max_date,
    value=(min_date, max_date), format="MMM YYYY",
    help="Scrubs the Overview trends, Compare, and Data tabs. Forecasting always trains on the full history.",
)
view_data = data.loc[view_start:view_end]
if view_data.empty:
    view_data = data

st.sidebar.markdown("---")
_jump_label = st.session_state.pop("indicator_jump", None)
_indicator_options = [label_lookup[c] for c in available_cols]
_default_idx = _indicator_options.index(_jump_label) if _jump_label in _indicator_options else 0
indicator_label = st.sidebar.selectbox(
    "Indicator to forecast", _indicator_options, index=_default_idx
)
indicator_col = INDICATORS[indicator_label]
if _jump_label:
    st.toast(f"Jumped to forecasting **{_jump_label}** — open the 🔮 Forecast tab", icon="🔮")

horizon = st.sidebar.slider("Forecast horizon (months)", 3, 24, 12)
model_name = st.sidebar.selectbox("Forecasting model", list(MODEL_FUNCS.keys()))

st.sidebar.markdown("---")
if data_source == "🔴 Live Data (auto-fetch)" and fetch_notes:
    with st.sidebar.expander("📡 Live source status", expanded=False):
        for col, note in fetch_notes.items():
            label = label_lookup.get(col, col)
            icon = "✅" if "unavailable" not in note and "no free" not in note else "⚠️"
            st.caption(f"{icon} **{label}** — {note}")
st.sidebar.caption(
    "⚠️ Live indicators update at different real-world frequencies "
    "(GDP/inflation/reserves are annual releases forward-filled monthly; "
    "FX & gold are daily). Treat forecasts as illustrative, not "
    "financial/policy advice."
)

# ----------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------
st.title("📊 Macroeconomic Performance Indicator & Predictor")
st.markdown(
    "AI-assisted forecasting for **GDP growth, inflation, forex reserves, "
    "gold price, currency volatility, FX rate, and policy rate** — built to help "
    "spot where the economy is heading next."
)

tab_overview, tab_forecast, tab_compare, tab_data = st.tabs(
    ["🏠 Overview & Risk", "🔮 Forecast", "🔍 Compare & Explore", "📄 Data"]
)

# ----------------------------------------------------------------------
# TAB 1 — OVERVIEW
# ----------------------------------------------------------------------
with tab_overview:
    stability_score, components = compute_risk_score(data)

    c1, c2 = st.columns([1, 2])
    with c1:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=stability_score,
                title={"text": "Economic Stability Score"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#1f77b4"},
                    "steps": [
                        {"range": [0, 33], "color": "#f8d7da"},
                        {"range": [33, 66], "color": "#fff3cd"},
                        {"range": [66, 100], "color": "#d4edda"},
                    ],
                },
            )
        )
        gauge.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(gauge, use_container_width=True)
        if components:
            with st.expander("What's driving this score?"):
                comp_df = pd.DataFrame(
                    {"Contribution": components}
                ).sort_values("Contribution")
                bar = go.Figure(go.Bar(
                    x=comp_df["Contribution"], y=comp_df.index, orientation="h",
                    marker_color=["#dc3545" if v < 0 else "#28a745" for v in comp_df["Contribution"]],
                ))
                bar.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10),
                                   xaxis_title="z-score contribution (+ = stabilizing)")
                st.plotly_chart(bar, use_container_width=True)

    with c2:
        st.subheader("Latest readings")
        st.caption("Click an indicator to jump straight to forecasting it →")
        latest = data.iloc[-1]
        cols = st.columns(3)
        for i, col in enumerate(available_cols):
            prev = data[col].iloc[-2] if len(data) > 1 else latest[col]
            delta = latest[col] - prev
            with cols[i % 3]:
                st.metric(label_lookup[col], f"{latest[col]:.2f}", f"{delta:+.2f}")
                if st.button("🔮 Forecast this", key=f"jump_{col}", use_container_width=True):
                    st.session_state["indicator_jump"] = label_lookup[col]
                    st.rerun()

    st.subheader(f"Indicator trends ({view_start.strftime('%b %Y')} – {view_end.strftime('%b %Y')})")
    recent = view_data
    n_cols_grid = 3
    n_rows_grid = int(np.ceil(len(available_cols) / n_cols_grid))
    fig = make_subplots(
        rows=n_rows_grid, cols=n_cols_grid,
        subplot_titles=[label_lookup[c] for c in available_cols],
    )
    positions = [(r + 1, c + 1) for r in range(n_rows_grid) for c in range(n_cols_grid)]
    for col, pos in zip(available_cols, positions):
        fig.add_trace(
            go.Scatter(x=recent.index, y=recent[col], mode="lines", name=label_lookup[col],
                       hovertemplate="%{x|%b %Y}: %{y:.2f}<extra></extra>"),
            row=pos[0], col=pos[1],
        )
    fig.update_layout(height=250 * n_rows_grid, showlegend=False, margin=dict(l=10, r=10, t=40, b=10),
                       hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Stability score blends z-scores of each indicator versus its own "
        "history — higher growth/reserves push it up, higher inflation/"
        "volatility/rate-instability pull it down. It's a heuristic "
        "snapshot, not a validated economic model. Use the date-range "
        "slider in the sidebar to scrub the trend charts above."
    )

# ----------------------------------------------------------------------
# TAB 2 — FORECAST
# ----------------------------------------------------------------------
with tab_forecast:
    st.subheader(f"Forecasting: {indicator_label}")
    series = data[indicator_col].dropna()

    opt1, opt2, opt3 = st.columns(3)
    show_band = opt1.checkbox("Show confidence band", value=True)
    log_scale_ok = (series > 0).all()
    log_scale = opt2.checkbox("Log scale", value=False, disabled=not log_scale_ok,
                               help=None if log_scale_ok else "Disabled — this series has zero/negative values.")
    show_compare = opt3.checkbox("Show all-model comparison inline", value=False)

    with st.expander("🧪 What-if scenario simulator", expanded=False):
        st.caption(
            "Shock the most recent reading and see how the forecast responds — "
            "useful for stress-testing (\"what if inflation jumps 10% right now?\")."
        )
        shock_pct = st.slider(
            f"Shock the latest {indicator_label} value by:",
            -30, 30, 0, format="%d%%", key="shock_slider",
        )

    model_func = MODEL_FUNCS[model_name]
    with st.spinner("Fitting model and generating forecast..."):
        try:
            preds, model_desc = model_func(series, horizon)
            mape = historical_backtest_error(series, model_func, min(horizon, 12))
        except Exception as e:
            st.error(f"Model failed to fit: {e}")
            st.stop()

    future_dates = pd.date_range(
        series.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"
    )
    forecast_series = pd.Series(preds, index=future_dates)

    resid_std = series.diff().std()
    upper = forecast_series + 1.96 * resid_std * np.sqrt(np.arange(1, horizon + 1) / 3)
    lower = forecast_series - 1.96 * resid_std * np.sqrt(np.arange(1, horizon + 1) / 3)

    shocked_forecast = None
    if shock_pct != 0:
        shocked_series = series.copy()
        shocked_series.iloc[-1] = shocked_series.iloc[-1] * (1 + shock_pct / 100)
        try:
            shocked_preds, _ = model_func(shocked_series, horizon)
            shocked_forecast = pd.Series(shocked_preds, index=future_dates)
        except Exception:
            shocked_forecast = None

    fig2 = go.Figure()
    hist_show = series.tail(48) if len(series) > 48 else series
    fig2.add_trace(go.Scatter(x=hist_show.index, y=hist_show.values, name="Historical",
                               line=dict(color="#1f77b4"),
                               hovertemplate="%{x|%b %Y}: %{y:.2f}<extra></extra>"))
    fig2.add_trace(go.Scatter(x=forecast_series.index, y=forecast_series.values, name="Forecast",
                               line=dict(color="#ff7f0e", dash="dash"),
                               hovertemplate="%{x|%b %Y}: %{y:.2f}<extra></extra>"))
    if show_band:
        fig2.add_trace(go.Scatter(
            x=list(forecast_series.index) + list(forecast_series.index[::-1]),
            y=list(upper.values) + list(lower.values[::-1]),
            fill="toself", fillcolor="rgba(255,127,14,0.15)",
            line=dict(color="rgba(255,255,255,0)"), name="~95% band", showlegend=True,
        ))
    if shocked_forecast is not None:
        fig2.add_trace(go.Scatter(
            x=shocked_forecast.index, y=shocked_forecast.values,
            name=f"Forecast w/ {shock_pct:+d}% shock",
            line=dict(color="#d62728", dash="dot"),
            hovertemplate="%{x|%b %Y}: %{y:.2f}<extra></extra>",
        ))
    fig2.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10),
                        xaxis_title="Date", yaxis_title=indicator_label,
                        hovermode="x unified",
                        yaxis_type="log" if (log_scale and log_scale_ok) else "linear")
    st.plotly_chart(fig2, use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model", model_desc)
    m2.metric("Current value", f"{series.iloc[-1]:.2f}")
    m3.metric(f"Forecast in {horizon}mo", f"{forecast_series.iloc[-1]:.2f}",
              f"{forecast_series.iloc[-1] - series.iloc[-1]:+.2f}")
    m4.metric("Backtest MAPE", f"{mape:.1f}%" if not np.isnan(mape) else "n/a",
              help="Mean absolute % error when the model is tested on the last known period it didn't see.")

    trend_dir = "rising 📈" if forecast_series.iloc[-1] > series.iloc[-1] else "falling 📉"
    st.info(f"**{indicator_label}** is trending **{trend_dir}** over the next {horizon} months under the {model_desc} model.")

    if show_compare:
        st.subheader("All models, this indicator")
        comp_fig = go.Figure()
        comp_fig.add_trace(go.Scatter(x=hist_show.index, y=hist_show.values, name="Historical", line=dict(color="#888")))
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        for (name, func), color in zip(MODEL_FUNCS.items(), palette):
            try:
                p, _ = func(series, horizon)
                comp_fig.add_trace(go.Scatter(x=future_dates, y=p, name=name, line=dict(color=color, dash="dash")))
            except Exception:
                pass
        comp_fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified")
        st.plotly_chart(comp_fig, use_container_width=True)

    with st.expander("Compare all models on this indicator (table)"):
        comp_rows = []
        for name, func in MODEL_FUNCS.items():
            try:
                p, _ = func(series, horizon)
                mp = historical_backtest_error(series, func, min(horizon, 12))
                comp_rows.append({"Model": name, f"Value in {horizon}mo": round(p[-1], 2), "Backtest MAPE %": mp})
            except Exception:
                comp_rows.append({"Model": name, f"Value in {horizon}mo": None, "Backtest MAPE %": None})
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download forecast as CSV",
        forecast_series.rename(indicator_col).to_csv().encode(),
        file_name=f"{indicator_col}_forecast.csv",
        mime="text/csv",
    )

# ----------------------------------------------------------------------
# TAB 3 — COMPARE & EXPLORE
# ----------------------------------------------------------------------
with tab_compare:
    st.subheader("Overlay indicators")
    default_pick = available_cols[: min(4, len(available_cols))]
    picked = st.multiselect(
        "Indicators to overlay",
        options=available_cols,
        default=default_pick,
        format_func=lambda c: label_lookup[c],
    )
    normalize = st.checkbox(
        "Normalize (z-score)", value=True,
        help="Puts every series on the same scale so shapes are comparable — raw units differ wildly (e.g. % vs USD Bn).",
    )

    if len(picked) >= 1:
        plot_df = view_data[picked].dropna(how="all")
        overlay = go.Figure()
        for col in picked:
            series_p = plot_df[col].dropna()
            if normalize and series_p.std() > 0:
                series_p = (series_p - series_p.mean()) / series_p.std()
            overlay.add_trace(go.Scatter(x=series_p.index, y=series_p.values, name=label_lookup[col],
                                          hovertemplate="%{x|%b %Y}: %{y:.2f}<extra></extra>"))
        overlay.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                               yaxis_title="z-score" if normalize else "raw value")
        st.plotly_chart(overlay, use_container_width=True)
    else:
        st.info("Pick at least one indicator above to plot it.")

    st.markdown("---")
    st.subheader("Correlation matrix")
    if len(available_cols) >= 2:
        corr_df = view_data[available_cols].dropna().corr()
        corr_df.index = [label_lookup[c] for c in corr_df.index]
        corr_df.columns = [label_lookup[c] for c in corr_df.columns]
        heat = px.imshow(
            corr_df, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1,
            aspect="auto",
        )
        heat.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(heat, use_container_width=True)
    else:
        st.info("Need at least two indicators with data to compute correlations.")

    st.markdown("---")
    st.subheader("Relationship explorer")
    if len(available_cols) >= 2:
        rc1, rc2 = st.columns(2)
        x_col = rc1.selectbox("X axis", available_cols, format_func=lambda c: label_lookup[c], index=0)
        remaining = [c for c in available_cols if c != x_col]
        y_col = rc2.selectbox("Y axis", remaining, format_func=lambda c: label_lookup[c], index=0)
        scatter_df = view_data[[x_col, y_col]].dropna().reset_index()
        if len(scatter_df) >= 3:
            try:
                scat = px.scatter(
                    scatter_df, x=x_col, y=y_col, trendline="ols",
                    labels={x_col: label_lookup[x_col], y_col: label_lookup[y_col]},
                    hover_data={scatter_df.columns[0]: True},
                )
            except Exception:
                scat = px.scatter(
                    scatter_df, x=x_col, y=y_col,
                    labels={x_col: label_lookup[x_col], y_col: label_lookup[y_col]},
                )
            scat.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(scat, use_container_width=True)
            corr_val = scatter_df[x_col].corr(scatter_df[y_col])
            st.caption(f"Pearson correlation: **{corr_val:.2f}** over the selected date range. Correlation isn't causation.")
        else:
            st.info("Not enough overlapping data points in the selected date range to plot this.")

# ----------------------------------------------------------------------
# TAB 4 — RAW DATA
# ----------------------------------------------------------------------
with tab_data:
    st.subheader("Underlying data")
    st.caption(f"Showing {view_start.strftime('%b %Y')} – {view_end.strftime('%b %Y')} (adjust the date range in the sidebar)")
    search_cols = st.multiselect(
        "Columns to show", available_cols, default=available_cols,
        format_func=lambda c: label_lookup[c], key="data_tab_cols",
    )
    st.dataframe(view_data[search_cols] if search_cols else view_data, use_container_width=True)
    st.download_button(
        "⬇️ Download full dataset as CSV",
        data.to_csv().encode(),
        file_name="macro_data.csv",
        mime="text/csv",
    )
