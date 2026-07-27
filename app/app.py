"""
Water Bottle Demand Forecasting - Interactive Dashboard
=======================================================

Run locally (e.g. from VS Code's integrated terminal):

    streamlit run app/app.py

The dashboard reads the artefacts produced by `run_pipeline.py`
(data/processed/*, models/*).  If they are missing it will tell you to run the
pipeline first.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `src` importable no matter where Streamlit is launched from.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.forecast import load_artifacts, predict_frame

# --------------------------------------------------------------------------- #
# Page config & theme
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Water Bottle Demand Forecasting",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Validated categorical palette (dataviz skill): blue / orange / aqua / violet.
REGION_COLORS = {
    "North_Zone": "#2a78d6",
    "South_Zone": "#eb6834",
    "East_Zone": "#1baf7a",
    "West_Zone": "#4a3aa7",
}
ACCENT = "#2a78d6"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
COLORWAY = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e87ba4", "#eda100"]

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 2rem;}
      .kpi-card {
        background: linear-gradient(135deg, rgba(42,120,214,0.10), rgba(27,175,122,0.08));
        border: 1px solid rgba(42,120,214,0.20);
        border-radius: 14px; padding: 18px 20px; height: 100%;
      }
      .kpi-label {font-size: 0.82rem; color: #6b7280; margin-bottom: 4px; font-weight: 600;
                  text-transform: uppercase; letter-spacing: .03em;}
      .kpi-value {font-size: 1.9rem; font-weight: 700; line-height: 1.1;}
      .kpi-sub {font-size: 0.8rem; color: #6b7280; margin-top: 2px;}
      h1, h2, h3 {letter-spacing: -0.01em;}
    </style>
    """,
    unsafe_allow_html=True,
)


def style_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    """Transparent, theme-adaptive Plotly styling."""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=COLORWAY,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(150,150,150,0.18)", zeroline=False)
    return fig


def kpi(col, label: str, value: str, sub: str = "") -> None:
    col.markdown(
        f"""<div class="kpi-card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value">{value}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Data loading (cached)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_history() -> pd.DataFrame:
    df = pd.read_csv(config.ENRICHED_FILE, parse_dates=["Date"])
    return df


@st.cache_data(show_spinner=False)
def load_forecasts() -> dict:
    return {
        "daily": pd.read_csv(config.FORECAST_DAILY_FILE, parse_dates=["Date"]),
        "weekly": pd.read_csv(config.FORECAST_WEEKLY_FILE, parse_dates=["Period"]),
        "monthly": pd.read_csv(config.FORECAST_MONTHLY_FILE, parse_dates=["Period"]),
        "yearly": pd.read_csv(config.FORECAST_YEARLY_FILE, parse_dates=["Period"]),
    }


@st.cache_data(show_spinner=False)
def load_geo() -> pd.DataFrame:
    return pd.read_csv(config.GEOGRAPHY_FILE)


@st.cache_resource(show_spinner=False)
def load_model():
    return load_artifacts()


def artefacts_exist() -> bool:
    needed = [
        config.ENRICHED_FILE, config.FORECAST_DAILY_FILE, config.MODEL_FILE,
        config.FEATURE_COLUMNS_FILE, config.METRICS_FILE, config.GEOGRAPHY_FILE,
    ]
    return all(p.exists() for p in needed)


if not artefacts_exist():
    st.title("💧 Water Bottle Demand Forecasting")
    st.error(
        "Model & data artefacts were not found.\n\n"
        "Run the pipeline once to generate them:\n\n"
        "```bash\npython run_pipeline.py\n```"
    )
    st.stop()

history = load_history()
fc = load_forecasts()
geo = load_geo()
model, feature_columns, metrics = load_model()

REGIONS = config.REGIONS
LAST_HIST_DATE = history["Date"].max()

# --------------------------------------------------------------------------- #
# Sidebar navigation
# --------------------------------------------------------------------------- #
st.sidebar.markdown("## 💧 Demand Forecasting")
st.sidebar.caption("Water-bottle manufacturing · multi-factor ML forecaster")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "📈 Historical Analysis", "🔮 Demand Forecast",
     "🗺️ Geographical View", "🧪 What-If Simulator", "🤖 Model & Data"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.metric("Model accuracy (100−MAPE)",
                  f"{100 - metrics['validation'][metrics['best_model']]['MAPE']:.1f}%")
st.sidebar.caption(
    f"History: {history['Date'].min().date()} → {LAST_HIST_DATE.date()}  ·  "
    f"{len(history):,} rows · {len(REGIONS)} zones"
)


# =========================================================================== #
# PAGE: OVERVIEW
# =========================================================================== #
def page_overview() -> None:
    st.title("📊 Executive Overview")
    st.caption("Historical demand at a glance, plus the next-year outlook.")

    total_hist = int(history["Target_Demand"].sum())
    avg_daily = history.groupby("Date")["Target_Demand"].sum().mean()
    peak_zone = history.groupby("Region")["Target_Demand"].mean().idxmax()
    next_year_total = int(fc["daily"]["Forecast_Demand"].sum())
    last_year_total = int(
        history[history["Date"].dt.year == LAST_HIST_DATE.year]["Target_Demand"].sum()
    )
    yoy = (next_year_total / last_year_total - 1) * 100

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Total demand (history)", f"{total_hist/1e6:.1f}M", "units, 2021–2023")
    kpi(c2, "Avg daily demand", f"{avg_daily/1e3:.1f}K", "units/day, all zones")
    kpi(c3, "Forecast next year", f"{next_year_total/1e6:.1f}M",
        f"{yoy:+.1f}% vs last year")
    kpi(c4, "Top zone", peak_zone.replace("_", " "),
        f"{history.groupby('Region')['Target_Demand'].mean()[peak_zone]/1e3:.1f}K avg/day")

    st.markdown("### Monthly demand — history & next-year forecast")
    hist_m = (
        history.assign(Period=history["Date"].dt.to_period("M").dt.to_timestamp())
        .groupby("Period")["Target_Demand"].sum().reset_index()
    )
    fc_m = fc["monthly"][fc["monthly"]["Region"] == "ALL"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_m["Period"], y=hist_m["Target_Demand"], name="Actual",
        mode="lines", line=dict(color=ACCENT, width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=fc_m["Period"], y=fc_m["Upper"], name="Upper", mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=fc_m["Period"], y=fc_m["Lower"], name="95% interval", mode="lines",
        line=dict(width=0), fill="tonexty",
        fillcolor="rgba(235,104,52,0.18)", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=fc_m["Period"], y=fc_m["Forecast_Demand"], name="Forecast",
        mode="lines", line=dict(color="#eb6834", width=2.5, dash="dot"),
    ))
    fig.add_vline(x=LAST_HIST_DATE.timestamp() * 1000, line_dash="dash",
                  line_color="rgba(150,150,150,0.6)")
    st.plotly_chart(style_fig(fig, 420), use_container_width=True)

    colA, colB = st.columns([1, 1])
    with colA:
        st.markdown("### Demand share by zone (history)")
        share = history.groupby("Region")["Target_Demand"].sum().reset_index()
        figd = px.pie(share, names="Region", values="Target_Demand", hole=0.55,
                      color="Region", color_discrete_map=REGION_COLORS)
        figd.update_traces(textposition="outside", textinfo="label+percent")
        st.plotly_chart(style_fig(figd, 360), use_container_width=True)
    with colB:
        st.markdown("### Top demand drivers")
        imp = pd.DataFrame(metrics["feature_importance"][:10])
        figi = px.bar(imp.sort_values("importance"), x="importance", y="feature",
                      orientation="h")
        figi.update_traces(marker_color=ACCENT)
        st.plotly_chart(style_fig(figi, 360), use_container_width=True)


# =========================================================================== #
# PAGE: HISTORICAL ANALYSIS
# =========================================================================== #
def page_history() -> None:
    st.title("📈 Historical Analysis")
    st.caption("Explore how weather, seasonality and geography shaped demand.")

    c1, c2 = st.columns([2, 1])
    sel_regions = c1.multiselect("Zones", REGIONS, default=REGIONS,
                                 format_func=lambda r: r.replace("_", " "))
    gran = c2.selectbox("Granularity", ["Daily", "Weekly", "Monthly"], index=2)
    if not sel_regions:
        st.info("Select at least one zone.")
        return

    d = history[history["Region"].isin(sel_regions)].copy()
    freq = {"Daily": "D", "Weekly": "W", "Monthly": "M"}[gran]
    d["Period"] = d["Date"].dt.to_period(freq).dt.start_time
    ts = d.groupby(["Period", "Region"])["Target_Demand"].sum().reset_index()

    st.markdown(f"### {gran} demand by zone")
    fig = px.line(ts, x="Period", y="Target_Demand", color="Region",
                  color_discrete_map=REGION_COLORS)
    fig.update_traces(line=dict(width=2))
    st.plotly_chart(style_fig(fig, 400), use_container_width=True)

    colA, colB = st.columns(2)
    with colA:
        st.markdown("### Demand by season")
        seas = (history[history["Region"].isin(sel_regions)]
                .groupby("Season")["Target_Demand"].mean()
                .reindex(["Winter", "Summer", "Monsoon", "Autumn"]).reset_index())
        figs = px.bar(seas, x="Season", y="Target_Demand")
        figs.update_traces(marker_color=ACCENT)
        figs.update_layout(hovermode="closest")
        st.plotly_chart(style_fig(figs, 340), use_container_width=True)
    with colB:
        st.markdown("### Temperature vs demand")
        samp = d.sample(min(1500, len(d)), random_state=1)
        figt = px.scatter(samp, x="Temperature_C", y="Target_Demand",
                          color="Region", color_discrete_map=REGION_COLORS,
                          opacity=0.55, trendline="lowess"
                          if _has_statsmodels() else None)
        figt.update_layout(hovermode="closest")
        st.plotly_chart(style_fig(figt, 340), use_container_width=True)

    st.markdown("### Correlation of demand with the enriched drivers")
    drivers = [
        "Target_Demand", "Temperature_C", "Heat_Index_C", "Humidity_pct",
        "Rainfall_mm", "Drought_Index", "Water_Stress_Index", "Air_Quality_Index",
        "Served_Population_M", "Urbanisation", "Economic_Index", "Tourism_Index",
        "Political_Stability_Index", "Is_Festival", "Plastic_Ban_Active",
    ]
    corr = history[drivers].corr()[["Target_Demand"]].drop("Target_Demand")
    corr = corr.sort_values("Target_Demand")
    figc = px.bar(corr, x="Target_Demand", y=corr.index, orientation="h",
                  color="Target_Demand", color_continuous_scale="RdBu",
                  range_color=[-1, 1])
    figc.update_layout(coloraxis_showscale=False, hovermode="closest")
    st.plotly_chart(style_fig(figc, 420), use_container_width=True)


def _has_statsmodels() -> bool:
    try:
        import statsmodels  # noqa: F401
        return True
    except Exception:
        return False


# =========================================================================== #
# PAGE: DEMAND FORECAST
# =========================================================================== #
def page_forecast() -> None:
    st.title("🔮 Demand Forecast")
    st.caption(f"Forward outlook from {(LAST_HIST_DATE + pd.Timedelta(days=1)).date()} "
               "using climatological scenarios for every driver.")

    c1, c2 = st.columns([1, 1])
    horizon = c1.radio("Horizon", ["Next Week", "Next Month", "Next Year"],
                       horizontal=True, index=2)
    zone = c2.selectbox("Zone", ["ALL (all zones)"] + REGIONS,
                        format_func=lambda r: r.replace("_", " "))
    zone_key = "ALL" if zone.startswith("ALL") else zone

    daily = fc["daily"].copy()
    if zone_key != "ALL":
        daily = daily[daily["Region"] == zone_key]
    daily = (daily.groupby("Date")[["Forecast_Demand", "Lower", "Upper"]]
             .sum().reset_index())

    if horizon == "Next Week":
        window = daily.head(7)
        agg = fc["daily"][fc["daily"]["Region"] == zone_key] if zone_key != "ALL" else None
        title = "Daily forecast — next 7 days"
    elif horizon == "Next Month":
        window = daily.head(31)
        title = "Daily forecast — next month"
    else:
        window = daily
        title = "Daily forecast — next year"

    total = int(window["Forecast_Demand"].sum())
    peak_day = window.loc[window["Forecast_Demand"].idxmax()]
    avg = window["Forecast_Demand"].mean()

    k1, k2, k3 = st.columns(3)
    kpi(k1, f"{horizon} total", f"{total:,}", "units")
    kpi(k2, "Avg / day", f"{avg:,.0f}", "units")
    kpi(k3, "Peak day", f"{peak_day['Forecast_Demand']:,.0f}",
        pd.to_datetime(peak_day["Date"]).strftime("%d %b %Y"))

    st.markdown(f"### {title}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=window["Date"], y=window["Upper"], mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=window["Date"], y=window["Lower"], mode="lines",
                             line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(42,120,214,0.16)", name="95% interval",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=window["Date"], y=window["Forecast_Demand"],
                             mode="lines", line=dict(color=ACCENT, width=2.5),
                             name="Forecast"))
    st.plotly_chart(style_fig(fig, 400), use_container_width=True)

    # Monthly / weekly breakdown table
    if horizon == "Next Year":
        st.markdown("### Monthly breakdown")
        table = fc["monthly"][fc["monthly"]["Region"] == zone_key].copy()
        table["Month"] = table["Period"].dt.strftime("%b %Y")
        show = table[["Month", "Forecast_Demand", "Lower", "Upper"]]
        figm = px.bar(table, x="Period", y="Forecast_Demand")
        figm.update_traces(marker_color=ACCENT)
        figm.update_layout(hovermode="closest")
        st.plotly_chart(style_fig(figm, 320), use_container_width=True)
    else:
        show = window.assign(Date=window["Date"].dt.strftime("%a %d %b %Y"))
        show = show[["Date", "Forecast_Demand", "Lower", "Upper"]]

    st.dataframe(show, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download this forecast (CSV)",
        window.to_csv(index=False).encode(),
        file_name=f"forecast_{horizon.lower().replace(' ', '_')}_{zone_key}.csv",
        mime="text/csv",
    )

    # Per-zone comparison for the full year
    st.markdown("### Forecast by zone (next year)")
    yr = fc["yearly"][fc["yearly"]["Region"] != "ALL"].copy()
    figz = px.bar(yr, x="Region", y="Forecast_Demand", color="Region",
                  color_discrete_map=REGION_COLORS)
    figz.update_layout(showlegend=False, hovermode="closest")
    st.plotly_chart(style_fig(figz, 320), use_container_width=True)


# =========================================================================== #
# PAGE: GEOGRAPHICAL VIEW
# =========================================================================== #
def page_geography() -> None:
    st.title("🗺️ Geographical View")
    st.caption("Zone locations, market attributes and demand density.")

    demand_by_zone = history.groupby("Region")["Target_Demand"].mean().reset_index()
    demand_by_zone.columns = ["Region", "Avg_Daily_Demand"]
    g = geo.merge(demand_by_zone, on="Region")
    fc_year = fc["yearly"][fc["yearly"]["Region"] != "ALL"][["Region", "Forecast_Demand"]]
    g = g.merge(fc_year, on="Region")

    fig = px.scatter_geo(
        g, lat="Latitude", lon="Longitude", color="Region",
        size="Avg_Daily_Demand", text="City_Proxy",
        color_discrete_map=REGION_COLORS, size_max=42, scope="asia",
        hover_data={"Latitude": False, "Longitude": False,
                    "Served_Population_M": True, "Climate_Zone": True,
                    "Avg_Daily_Demand": ":,.0f"},
    )
    fig.update_traces(textposition="top center")
    fig.update_geos(center=dict(lat=22, lon=80), projection_scale=3.2,
                    showcountries=True, landcolor="rgba(150,150,150,0.10)",
                    countrycolor="rgba(150,150,150,0.35)")
    fig.update_layout(height=460, margin=dict(l=0, r=0, t=10, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", showlegend=True,
                      legend=dict(orientation="h", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    colA, colB = st.columns([3, 2])
    with colA:
        st.markdown("### Zone attributes")
        st.dataframe(
            g[["Region", "City_Proxy", "Climate_Zone", "Served_Population_M",
               "Urbanisation", "Elevation_m", "Coastal", "Avg_Daily_Demand",
               "Forecast_Demand"]].rename(columns={"Forecast_Demand": "Next_Year_Forecast"}),
            use_container_width=True, hide_index=True,
        )
    with colB:
        st.markdown("### Market size vs demand")
        figp = px.scatter(g, x="Served_Population_M", y="Avg_Daily_Demand",
                          color="Region", text="Region",
                          color_discrete_map=REGION_COLORS, size="Avg_Daily_Demand",
                          size_max=40)
        figp.update_traces(textposition="top center")
        figp.update_layout(showlegend=False, hovermode="closest")
        st.plotly_chart(style_fig(figp, 360), use_container_width=True)


# =========================================================================== #
# PAGE: WHAT-IF SIMULATOR
# =========================================================================== #
def _baseline_row(region: str, month: int) -> pd.DataFrame:
    """Representative feature row for a zone & month, from historical climatology."""
    sub = history[(history["Region"] == region) & (history["Date"].dt.month == month)]
    if sub.empty:
        sub = history[history["Region"] == region]
    num = sub.select_dtypes(include=[np.number]).mean()
    row = num.to_frame().T
    row["Region"] = region
    row["Season"] = config.MONTH_TO_SEASON[month]
    row["Climate_Zone"] = geo.loc[geo["Region"] == region, "Climate_Zone"].iloc[0]
    row["City_Proxy"] = geo.loc[geo["Region"] == region, "City_Proxy"].iloc[0]
    row["Date"] = pd.Timestamp(2024, month, 15)
    return row


def page_whatif() -> None:
    st.title("🧪 What-If Scenario Simulator")
    st.caption("Move the levers and watch predicted daily demand respond — "
               "powered by the same model that produces the forecast.")

    left, right = st.columns([1, 1.4])
    with left:
        region = st.selectbox("Zone", REGIONS, format_func=lambda r: r.replace("_", " "))
        month = st.select_slider(
            "Month", options=list(range(1, 13)),
            format_func=lambda m: pd.Timestamp(2024, m, 1).strftime("%B"), value=6,
        )
        base = _baseline_row(region, month)

        st.markdown("**Weather & climate**")
        temp = st.slider("Temperature (°C)", 5.0, 50.0,
                         float(round(base["Temperature_C"].iloc[0], 1)), 0.5)
        humidity = st.slider("Humidity (%)", 10, 99,
                             int(base["Humidity_pct"].iloc[0]))
        drought = st.slider("Drought index", 0.0, 1.0,
                            float(round(base["Drought_Index"].iloc[0], 2)), 0.01)
        water_stress = st.slider("Water-stress index", 0.0, 1.0,
                                 float(round(base["Water_Stress_Index"].iloc[0], 2)), 0.01)

        st.markdown("**Economy & policy**")
        econ = st.slider("Economic index", 90.0, 130.0,
                         float(round(base["Economic_Index"].iloc[0], 1)), 0.5)
        tourism = st.slider("Tourism index", 10, 100,
                            int(base["Tourism_Index"].iloc[0]))
        festival = st.toggle("Festival day", value=False)
        plastic_ban = st.toggle("Plastic-bottle ban active",
                                value=bool(base["Plastic_Ban_Active"].iloc[0] >= 0.5))
        election = st.toggle("Election period", value=False)
        extreme = st.toggle("Extreme-weather event", value=False)

    # Build scenario row
    scen = base.copy()
    from src.data_generation import _heat_index
    scen["Temperature_C"] = temp
    scen["Humidity_pct"] = humidity
    scen["Heat_Index_C"] = float(_heat_index(np.array([temp]), np.array([humidity]))[0])
    scen["Drought_Index"] = drought
    scen["Water_Stress_Index"] = water_stress
    scen["Economic_Index"] = econ
    scen["Tourism_Index"] = tourism
    scen["Is_Festival"] = int(festival)
    scen["Plastic_Ban_Active"] = int(plastic_ban)
    scen["Election_Period"] = int(election)
    scen["Extreme_Weather_Flag"] = int(extreme)

    pred = float(predict_frame(scen, model, feature_columns)[0])
    base_pred = float(predict_frame(base, model, feature_columns)[0])
    delta = pred - base_pred
    pct = (delta / base_pred * 100) if base_pred else 0

    with right:
        st.markdown("### Predicted daily demand")
        st.markdown(
            f"<div style='font-size:3.4rem;font-weight:800;color:{ACCENT};line-height:1'>"
            f"{pred:,.0f}</div><div style='color:#6b7280'>units / day</div>",
            unsafe_allow_html=True,
        )
        color = GOOD if delta >= 0 else CRITICAL
        arrow = "▲" if delta >= 0 else "▼"
        st.markdown(
            f"<div style='margin-top:8px;font-size:1.1rem;color:{color};font-weight:600'>"
            f"{arrow} {delta:+,.0f} units ({pct:+.1f}%) vs typical {region.replace('_',' ')} "
            f"in {pd.Timestamp(2024, month, 1).strftime('%B')}</div>",
            unsafe_allow_html=True,
        )

        # Sensitivity to temperature at the current settings
        st.markdown("### Sensitivity to temperature")
        temps = np.arange(5, 50.5, 1.0)
        grid = pd.concat([scen] * len(temps), ignore_index=True)
        grid["Temperature_C"] = temps
        grid["Heat_Index_C"] = _heat_index(temps, np.full(len(temps), humidity))
        grid_pred = predict_frame(grid, model, feature_columns)
        figs = go.Figure()
        figs.add_trace(go.Scatter(x=temps, y=grid_pred, mode="lines",
                                  line=dict(color=ACCENT, width=2.5), name="Demand"))
        figs.add_vline(x=temp, line_dash="dash", line_color=CRITICAL)
        figs.update_layout(hovermode="x")
        st.plotly_chart(style_fig(figs, 320), use_container_width=True)

    st.info("Every lever above is one of the enriched climate / weather / "
            "economic / policy features. The model learned their combined effect "
            "from 3 years of daily data across 4 zones.")


# =========================================================================== #
# PAGE: MODEL & DATA
# =========================================================================== #
def page_model() -> None:
    st.title("🤖 Model & Data")
    st.caption("How the forecaster was built, how accurate it is, and what fed it.")

    best = metrics["best_model"]
    val = metrics["validation"][best]
    cv = metrics.get("cross_validation", {})

    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "Best model", best.replace("Gradient", "GB"), "auto-selected")
    kpi(k2, "MAE (hold-out)", f"{val['MAE']:,.0f}", "units")
    kpi(k3, "MAPE", f"{val['MAPE']:.1f}%", f"≈ {100-val['MAPE']:.1f}% accuracy")
    kpi(k4, "R²", f"{val['R2']:.3f}", "explained variance")

    st.markdown("### Validation strategy")
    st.markdown(
        f"- **Hold-out:** trained on data before `{metrics['holdout_split_date']}`, "
        f"tested on the most recent year ({metrics['n_val_rows']:,} rows).\n"
        f"- **Rolling CV:** 5-fold time-series cross-validation — "
        f"MAE **{cv.get('MAE', float('nan')):,.0f}**, MAPE **{cv.get('MAPE', float('nan')):.1f}%**, "
        f"R² **{cv.get('R2', float('nan')):.3f}**.\n"
        f"- **Features:** {metrics['n_features']} predictors spanning weather, "
        f"climate, geography, economy and policy."
    )

    colA, colB = st.columns([1, 1])
    with colA:
        st.markdown("### Feature importance (top 15)")
        imp = pd.DataFrame(metrics["feature_importance"][:15])
        fig = px.bar(imp.sort_values("importance"), x="importance", y="feature",
                     orientation="h")
        fig.update_traces(marker_color=ACCENT)
        fig.update_layout(hovermode="closest")
        st.plotly_chart(style_fig(fig, 480), use_container_width=True)
    with colB:
        st.markdown("### Actual vs predicted (hold-out year)")
        val_df = _holdout_actual_vs_pred()
        figp = px.scatter(val_df, x="Actual", y="Predicted", opacity=0.4)
        figp.update_traces(marker=dict(color=ACCENT, size=5))
        lim = [val_df[["Actual", "Predicted"]].min().min(),
               val_df[["Actual", "Predicted"]].max().max()]
        figp.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
                                  line=dict(color=CRITICAL, dash="dash"),
                                  name="perfect"))
        figp.update_layout(hovermode="closest", showlegend=False)
        st.plotly_chart(style_fig(figp, 480), use_container_width=True)

    st.markdown("### The enriched dataset")
    st.markdown(
        "The provided demand series was enriched with four generated datasets — "
        "**geography**, **weather**, **climate** and **political/economic** — into a "
        f"single modelling table of **{history.shape[0]:,} rows × {history.shape[1]} columns**."
    )
    st.dataframe(history.head(12), use_container_width=True, hide_index=True)

    files = {
        "Enriched (master)": config.ENRICHED_FILE,
        "Geography": config.GEOGRAPHY_FILE,
        "Weather": config.WEATHER_FILE,
        "Climate": config.CLIMATE_FILE,
        "Political/Economic": config.POLITICAL_FILE,
    }
    st.markdown("### Download the datasets")
    cols = st.columns(len(files))
    for col, (label, path) in zip(cols, files.items()):
        if path.exists():
            col.download_button(f"⬇️ {label}", path.read_bytes(),
                                file_name=path.name, mime="text/csv")


@st.cache_data(show_spinner=False)
def _holdout_actual_vs_pred() -> pd.DataFrame:
    val = history[history["Date"] >= pd.Timestamp(metrics["holdout_split_date"])]
    pred = predict_frame(val, model, feature_columns)
    return pd.DataFrame({"Actual": val["Target_Demand"].values, "Predicted": pred})


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
PAGES = {
    "📊 Overview": page_overview,
    "📈 Historical Analysis": page_history,
    "🔮 Demand Forecast": page_forecast,
    "🗺️ Geographical View": page_geography,
    "🧪 What-If Simulator": page_whatif,
    "🤖 Model & Data": page_model,
}
PAGES[page]()

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit · scikit-learn · Plotly")
