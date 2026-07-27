"""
Future forecasting.

The trained model is a multi-factor regressor, so to forecast the future we
must first build plausible future values for every driver.  We do that with
**climatological normals**: for each zone and calendar day we take the average
historical weather / climate / macro conditions, then project the trended
signals (temperature warming, economic growth) forward.  Festivals are taken
from the recurring festival calendar and any active plastic-ban policy is
carried forward.

The model then scores those future rows to produce a daily demand forecast,
which is aggregated into next-week, next-month and next-year views.

Run with:  python -m src.forecast
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from . import config
from .features import make_model_matrix

# Features we fill from historical seasonal climatology (by region + calendar day)
CLIMATOLOGY_FEATURES = [
    "Temperature_C", "Humidity_pct", "Rainfall_mm", "Heat_Index_C",
    "Wind_Speed_kmph", "UV_Index", "Drought_Index", "Air_Quality_Index",
    "Water_Stress_Index", "Temp_Anomaly_C", "Political_Stability_Index",
    "Economic_Index", "Consumer_Confidence", "Tourism_Index", "Water_Tax_Rate",
]
# Assumed absent in the central ("expected") scenario.
ZERO_BASELINE = ["Extreme_Weather_Flag", "Election_Period", "Govt_Subsidy_Active"]
# Policy state carried forward from the last known day per region.
CARRY_FORWARD = ["Plastic_Ban_Active"]
# Annual trends applied on top of the (2022-centred) climatology.
TREND_PER_YEAR = {
    "Temperature_C": 0.15,
    "Temp_Anomaly_C": 0.15,
    "Economic_Index": 4.0,
    "Consumer_Confidence": 1.6,
    "Water_Tax_Rate": 0.25,
}
_BASELINE_YEAR = 2022  # midpoint of the 2021-2023 history


# --------------------------------------------------------------------------- #
# Loading helpers (also used by the dashboard)
# --------------------------------------------------------------------------- #
def load_artifacts():
    model = joblib.load(config.MODEL_FILE)
    feature_columns = json.loads(config.FEATURE_COLUMNS_FILE.read_text())
    metrics = json.loads(config.METRICS_FILE.read_text())
    return model, feature_columns, metrics


def predict_frame(df: pd.DataFrame, model=None, feature_columns=None) -> np.ndarray:
    """Score an already-enriched dataframe (used by the what-if simulator)."""
    if model is None or feature_columns is None:
        model, feature_columns, _ = load_artifacts()
    X, _ = make_model_matrix(df, feature_columns=feature_columns)
    return np.clip(model.predict(X), 0, None)


# --------------------------------------------------------------------------- #
# Build future exogenous feature rows
# --------------------------------------------------------------------------- #
def build_future_frame(history: pd.DataFrame, days: int = config.FORECAST_DAYS) -> pd.DataFrame:
    history = history.copy()
    history["Date"] = pd.to_datetime(history["Date"])
    history["Month"] = history["Date"].dt.month
    history["Day"] = history["Date"].dt.day

    geo = pd.read_csv(config.GEOGRAPHY_FILE)
    festivals = pd.read_csv(config.FESTIVAL_FILE)
    festival_md = set(zip(festivals["Month"], festivals["Day"]))

    # Seasonal climatology per (Region, Month, Day).
    clim = (
        history.groupby(["Region", "Month", "Day"])[CLIMATOLOGY_FEATURES]
        .mean()
        .reset_index()
    )

    # Last known policy state per region.
    last_day = history.sort_values("Date").groupby("Region").tail(1)
    carry = last_day.set_index("Region")[CARRY_FORWARD]

    start = history["Date"].max() + pd.Timedelta(days=1)
    future_dates = pd.date_range(start=start, periods=days, freq="D")

    rows = []
    for region in config.REGIONS:
        fdf = pd.DataFrame({"Date": future_dates})
        fdf["Region"] = region
        fdf["Month"] = fdf["Date"].dt.month
        fdf["Day"] = fdf["Date"].dt.day
        fdf["Year"] = fdf["Date"].dt.year
        fdf["Season"] = fdf["Month"].map(config.MONTH_TO_SEASON)

        # Merge seasonal climatology.
        fdf = fdf.merge(
            clim[clim["Region"] == region].drop(columns="Region"),
            on=["Month", "Day"], how="left",
        )
        # Fill leap-day (Feb 29) gaps with neighbouring days.
        fdf = fdf.sort_values("Date")
        fdf[CLIMATOLOGY_FEATURES] = fdf[CLIMATOLOGY_FEATURES].ffill().bfill()

        # Apply forward trends.
        years_ahead = fdf["Year"] - _BASELINE_YEAR
        for col, slope in TREND_PER_YEAR.items():
            fdf[col] = fdf[col] + slope * years_ahead

        # Zero-baseline (no assumed shocks in the expected scenario).
        for col in ZERO_BASELINE:
            fdf[col] = 0
        # Carry-forward policy state.
        for col in CARRY_FORWARD:
            fdf[col] = int(carry.loc[region, col])
        # Recurring festivals.
        fdf["Is_Festival"] = [
            int((m, d) in festival_md) for m, d in zip(fdf["Month"], fdf["Day"])
        ]

        rows.append(fdf)

    future = pd.concat(rows, ignore_index=True)
    # Attach static geography (lat/lon, market size, climate zone, ...).
    future = future.merge(geo, on="Region", how="left")
    return future


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _aggregate(daily: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Aggregate daily forecast to weekly/monthly/yearly, per region + ALL."""
    d = daily.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    if freq == "W":
        d["Period"] = d["Date"].dt.to_period("W").apply(lambda p: p.start_time)
    elif freq == "M":
        d["Period"] = d["Date"].dt.to_period("M").dt.to_timestamp()
    elif freq == "Y":
        d["Period"] = d["Date"].dt.to_period("Y").dt.to_timestamp()
    else:
        raise ValueError(freq)

    by_region = (
        d.groupby(["Period", "Region"])[["Forecast_Demand", "Lower", "Upper"]]
        .sum().reset_index()
    )
    total = (
        d.groupby("Period")[["Forecast_Demand", "Lower", "Upper"]]
        .sum().reset_index()
    )
    total["Region"] = "ALL"
    out = pd.concat([by_region, total], ignore_index=True)
    return out.sort_values(["Region", "Period"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_forecast(save: bool = True) -> dict:
    model, feature_columns, metrics = load_artifacts()
    residual_std = metrics.get("residual_std", 0.0)

    history = pd.read_csv(config.ENRICHED_FILE, parse_dates=["Date"])
    future = build_future_frame(history)

    preds = predict_frame(future, model, feature_columns)
    z = 1.96
    daily = pd.DataFrame({
        "Date": future["Date"].values,
        "Region": future["Region"].values,
        "Forecast_Demand": np.round(preds).astype(int),
        "Lower": np.round(np.clip(preds - z * residual_std, 0, None)).astype(int),
        "Upper": np.round(preds + z * residual_std).astype(int),
        "Temperature_C": future["Temperature_C"].round(1).values,
        "Season": future["Season"].values,
        "Is_Festival": future["Is_Festival"].values,
        "Plastic_Ban_Active": future["Plastic_Ban_Active"].values,
    })

    weekly = _aggregate(daily, "W")
    monthly = _aggregate(daily, "M")
    yearly = _aggregate(daily, "Y")

    if save:
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        daily.to_csv(config.FORECAST_DAILY_FILE, index=False)
        weekly.to_csv(config.FORECAST_WEEKLY_FILE, index=False)
        monthly.to_csv(config.FORECAST_MONTHLY_FILE, index=False)
        yearly.to_csv(config.FORECAST_YEARLY_FILE, index=False)
        horizon = f"{daily['Date'].min().date()} -> {daily['Date'].max().date()}"
        total_year = int(daily["Forecast_Demand"].sum())
        print(f"[forecast] horizon: {horizon} ({len(future)} region-days)")
        print(f"[forecast] next-year total demand (all zones): {total_year:,} units")
        print(f"[forecast] saved daily/weekly/monthly/yearly forecasts to {config.PROCESSED_DIR}")

    return {
        "daily": daily, "weekly": weekly, "monthly": monthly, "yearly": yearly,
    }


if __name__ == "__main__":
    run_forecast(save=True)
