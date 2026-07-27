"""
Data generation / enrichment layer.

The brief asks us to "create more datasets regarding climate, political
condition, weather and geographical location" and then forecast demand using
all of it.  This module takes the *provided* daily demand dataset as the
ground-truth backbone and builds four additional, physically-plausible
datasets around it:

    1. geography.csv            - static per-zone attributes (lat/lon, market
                                  size, urbanisation, elevation, coast, climate
                                  zone) used for the map + regional baselines.
    2. weather.csv              - daily weather per zone (humidity, rainfall,
                                  heat-index, wind, UV, extreme-weather flag)
                                  derived from the given temperature + season.
    3. climate.csv              - slower-moving climate signals (warming
                                  anomaly, drought index, air quality, water
                                  stress).
    4. political_economic.csv   - policy / macro signals (plastic-bottle ban,
                                  political stability, economic index, consumer
                                  confidence, tourism, elections, water tax).

Everything is merged into `enriched_demand.csv`.  The modelling target
`Target_Demand` is the provided demand *adjusted* by a small, transparent and
fully documented set of multiplicative effects from the new drivers (e.g. a
plastic-bottle ban shaves ~10% off demand).  The original untouched value is
preserved as `Base_Demand`, and the two stay >0.95 correlated, so the data
remains anchored to what you supplied while every new feature carries genuine,
learnable signal that the dashboard "what-if" levers can move.

Run directly with:  python -m src.data_generation
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------- #
# 1. Geography (static per zone)
# --------------------------------------------------------------------------- #
# Coordinates are real Indian metro proxies for each distribution zone so the
# dashboard map is meaningful.  `Served_Population_M` and `Urbanisation` are
# calibrated to the demand baselines in the provided data (North & South zones
# serve larger, more urban markets and show higher demand).
GEOGRAPHY = [
    # region,       city proxy, lat,    lon,    served_pop_M, urban, elev_m, coastal, area_k_km2, climate_zone
    ("North_Zone", "Delhi NCR", 28.61, 77.21, 30.5, 0.92, 216, 0, 1.48, "Semi_Arid"),
    ("South_Zone", "Chennai",   13.08, 80.27, 29.2, 0.90, 6,   1, 0.43, "Tropical_Wet_Dry"),
    ("East_Zone",  "Kolkata",   22.57, 88.36, 18.4, 0.82, 9,   1, 1.85, "Humid_Subtropical"),
    ("West_Zone",  "Mumbai",    19.08, 72.88, 17.1, 0.83, 14,  1, 0.60, "Tropical_Monsoon"),
]


def build_geography() -> pd.DataFrame:
    cols = [
        "Region", "City_Proxy", "Latitude", "Longitude", "Served_Population_M",
        "Urbanisation", "Elevation_m", "Coastal", "Area_k_km2", "Climate_Zone",
    ]
    return pd.DataFrame(GEOGRAPHY, columns=cols)


# --------------------------------------------------------------------------- #
# 2. Weather (daily, derived from the given temperature + season)
# --------------------------------------------------------------------------- #
def _heat_index(temp_c: np.ndarray, humidity: np.ndarray) -> np.ndarray:
    """Simplified NOAA heat index (a.k.a. 'feels like') in Celsius."""
    t = temp_c * 9 / 5 + 32  # to Fahrenheit
    rh = humidity
    hi = (
        -42.379 + 2.04901523 * t + 10.14333127 * rh
        - 0.22475541 * t * rh - 6.83783e-3 * t ** 2
        - 5.481717e-2 * rh ** 2 + 1.22874e-3 * t ** 2 * rh
        + 8.5282e-4 * t * rh ** 2 - 1.99e-6 * t ** 2 * rh ** 2
    )
    # Heat index only meaningful above ~26.6C / 80F; fall back to temp below that
    hi = np.where(t < 80, t, hi)
    return (hi - 32) * 5 / 9  # back to Celsius


def build_weather(base: pd.DataFrame, geo: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = base.merge(geo[["Region", "Coastal"]], on="Region", how="left")
    n = len(df)
    season = df["Season"].values
    temp = df["Temperature_C"].values
    coastal = df["Coastal"].values

    # Humidity: coastal + monsoon push it up, hot dry summers pull it down.
    base_hum = 52 + 15 * coastal
    season_hum = np.select(
        [season == "Monsoon", season == "Winter", season == "Summer", season == "Autumn"],
        [28, 6, -18, 4], default=0.0,
    )
    humidity = base_hum + season_hum - 0.6 * (temp - 25) + rng.normal(0, 5, n)
    humidity = np.clip(humidity, 12, 99)

    # Rainfall: concentrated in the monsoon, heavier on the coast.
    monsoon = (season == "Monsoon").astype(float)
    rain_lambda = 2 + monsoon * (14 + 10 * coastal)
    rainfall = rng.gamma(shape=1.1, scale=rain_lambda) * (rng.random(n) < (0.15 + 0.6 * monsoon))
    rainfall = np.round(rainfall, 1)

    heat_index = np.round(_heat_index(temp, humidity), 1)

    wind = np.clip(6 + 8 * monsoon + rng.normal(0, 3, n), 1, 45).round(1)

    uv = np.select(
        [season == "Summer", season == "Monsoon", season == "Winter", season == "Autumn"],
        [9.0, 5.5, 4.0, 6.5], default=6.0,
    ) + 0.05 * (temp - 25) + rng.normal(0, 0.6, n)
    uv = np.clip(uv, 0, 12).round(1)

    # Extreme weather: heatwaves (very hot) or flooding (very wet).
    extreme = ((temp > 44) | (rainfall > 60)).astype(int)

    out = pd.DataFrame({
        "Date": df["Date"].values,
        "Region": df["Region"].values,
        "Humidity_pct": humidity.round(1),
        "Rainfall_mm": rainfall,
        "Heat_Index_C": heat_index,
        "Wind_Speed_kmph": wind,
        "UV_Index": uv,
        "Extreme_Weather_Flag": extreme,
    })
    return out


# --------------------------------------------------------------------------- #
# 3. Climate (slower moving signals)
# --------------------------------------------------------------------------- #
def build_climate(base: pd.DataFrame, geo: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = base.merge(
        geo[["Region", "Climate_Zone", "Coastal", "Elevation_m"]], on="Region", how="left"
    )
    n = len(df)
    year = df["Date"].dt.year.values
    season = df["Season"].values
    temp = df["Temperature_C"].values

    # Long-term warming anomaly relative to the 2021 baseline (~+0.15C / yr).
    temp_anomaly = (year - 2021) * 0.15 + rng.normal(0, 0.1, n)

    # Drought index (0-1): high in dry summers, near zero in the monsoon,
    # structurally higher in the semi-arid North.
    arid_boost = np.where(df["Climate_Zone"].values == "Semi_Arid", 0.18, 0.0)
    drought = np.select(
        [season == "Summer", season == "Monsoon", season == "Winter", season == "Autumn"],
        [0.62, 0.08, 0.42, 0.38], default=0.35,
    ) + arid_boost + rng.normal(0, 0.06, n)
    drought = np.clip(drought, 0, 1).round(3)

    # Air Quality Index: worst in winter, especially the landlocked North.
    aqi = np.select(
        [season == "Winter", season == "Autumn", season == "Summer", season == "Monsoon"],
        [150, 120, 95, 60], default=90,
    ) + np.where(df["Climate_Zone"].values == "Semi_Arid", 45, 0) + rng.normal(0, 18, n)
    aqi = np.clip(aqi, 20, 400).round(0)

    # Water-stress index: regional baseline lifted by drought.
    stress_base = df["Region"].map(
        {"North_Zone": 0.70, "South_Zone": 0.52, "East_Zone": 0.44, "West_Zone": 0.56}
    ).values
    water_stress = np.clip(stress_base + 0.25 * (drought - 0.4) + rng.normal(0, 0.03, n), 0, 1).round(3)

    return pd.DataFrame({
        "Date": df["Date"].values,
        "Region": df["Region"].values,
        "Climate_Zone": df["Climate_Zone"].values,
        "Temp_Anomaly_C": temp_anomaly.round(3),
        "Drought_Index": drought,
        "Air_Quality_Index": aqi,
        "Water_Stress_Index": water_stress,
    })


# --------------------------------------------------------------------------- #
# 4. Political / economic conditions
# --------------------------------------------------------------------------- #
# Plastic-bottle ban roll-out schedule (a real, learnable policy lever).
PLASTIC_BAN_SCHEDULE = {
    "West_Zone": pd.Timestamp("2022-07-01"),
    "North_Zone": pd.Timestamp("2022-10-01"),
    # South_Zone and East_Zone have no ban in the history (0), but the
    # dashboard can still toggle the lever to simulate one.
}

# Election periods (public gatherings lift short-term demand).
ELECTION_WINDOWS = [
    (pd.Timestamp("2022-02-01"), pd.Timestamp("2022-03-15")),
    (pd.Timestamp("2023-05-01"), pd.Timestamp("2023-05-31")),
]


def build_political(base: pd.DataFrame, geo: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = base[["Date", "Region", "Season"]].copy()
    n = len(df)
    dates = df["Date"]
    year = dates.dt.year.values
    doy = dates.dt.dayofyear.values
    season = df["Season"].values

    # Economic index: ~100 in 2021 growing ~4/yr with a mid-2022 slowdown.
    econ = 100 + (year - 2021) * 4 + 2 * np.sin(2 * np.pi * doy / 365)
    slowdown = ((dates >= "2022-05-01") & (dates <= "2022-10-01")).values
    econ = econ - slowdown * 3 + rng.normal(0, 1.0, n)
    econ = econ.round(2)

    consumer_conf = np.clip(55 + 0.4 * (econ - 105) + rng.normal(0, 4, n), 20, 95).round(1)

    # Tourism: peaks in the cool winter, dips in the monsoon, festival bumps.
    tourism = np.select(
        [season == "Winter", season == "Autumn", season == "Summer", season == "Monsoon"],
        [72, 60, 45, 34], default=50,
    ) + base["Is_Festival"].values * 12 + rng.normal(0, 5, n)
    tourism = np.clip(tourism, 10, 100).round(1)

    # Political stability: mostly steady per zone with small drift.
    stab_base = df["Region"].map(
        {"North_Zone": 78, "South_Zone": 80, "East_Zone": 74, "West_Zone": 82}
    ).values
    political_stability = np.clip(stab_base + rng.normal(0, 3, n), 40, 100).round(1)

    # Plastic ban flag from the roll-out schedule.
    plastic_ban = np.zeros(n, dtype=int)
    for region, start in PLASTIC_BAN_SCHEDULE.items():
        mask = (df["Region"] == region).values & (dates >= start).values
        plastic_ban[mask] = 1

    # Election flag.
    election = np.zeros(n, dtype=int)
    for start, end in ELECTION_WINDOWS:
        election[((dates >= start) & (dates <= end)).values] = 1

    # Water tax (%) and subsidy programmes.
    water_tax = df["Region"].map(
        {"North_Zone": 5.0, "South_Zone": 4.0, "East_Zone": 3.5, "West_Zone": 6.0}
    ).values + (year - 2021) * 0.25
    subsidy = ((season == "Summer") & (rng.random(n) < 0.5)).astype(int)

    return pd.DataFrame({
        "Date": df["Date"].values,
        "Region": df["Region"].values,
        "Political_Stability_Index": political_stability,
        "Plastic_Ban_Active": plastic_ban,
        "Economic_Index": econ,
        "Consumer_Confidence": consumer_conf,
        "Tourism_Index": tourism,
        "Election_Period": election,
        "Water_Tax_Rate": np.round(water_tax, 2),
        "Govt_Subsidy_Active": subsidy,
    })


# --------------------------------------------------------------------------- #
# Festival calendar (extracted from the data so the forecaster can reuse it)
# --------------------------------------------------------------------------- #
def build_festival_calendar(base: pd.DataFrame) -> pd.DataFrame:
    fest = base[base["Is_Festival"] == 1][["Date"]].drop_duplicates().copy()
    fest["Month"] = fest["Date"].dt.month
    fest["Day"] = fest["Date"].dt.day
    fest = fest.sort_values("Date").reset_index(drop=True)
    return fest


# --------------------------------------------------------------------------- #
# Demand enrichment
# --------------------------------------------------------------------------- #
def apply_demand_enrichment(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Adjust the provided demand by transparent multiplicative effects from the
    new drivers.  Effects are centred on 1.0 and bounded, so the enriched
    `Target_Demand` stays realistic and highly correlated with the original
    `Base_Demand`, while giving every driver real, learnable signal.
    """
    base = df["Base_Demand"].astype(float).values

    mult = np.ones(len(df))
    # Policy: a plastic-bottle ban shaves ~10% off single-use bottle demand.
    mult *= 1 - 0.10 * df["Plastic_Ban_Active"].values
    # Macro: a stronger economy lifts discretionary consumption.
    mult *= 1 + 0.0015 * (df["Economic_Index"].values - 105)
    # Climate: water-stressed zones lean harder on bottled water.
    mult *= 1 + 0.12 * (df["Water_Stress_Index"].values - 0.5)
    # Tourism: more visitors, more on-the-go bottle purchases.
    mult *= 1 + 0.0008 * (df["Tourism_Index"].values - 50)
    # Short-lived drought spikes.
    mult *= 1 + 0.06 * (df["Drought_Index"].values - 0.4)
    # Public gatherings during elections and extreme heat events.
    mult *= 1 + 0.05 * df["Election_Period"].values
    mult *= 1 + 0.05 * df["Extreme_Weather_Flag"].values

    noise = rng.normal(0, 0.015, len(df))
    target = base * mult * (1 + noise)
    df["Target_Demand"] = np.clip(np.round(target), 0, None).astype(int)
    return df


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def generate_all(save: bool = True) -> pd.DataFrame:
    """Build every external dataset, merge, enrich and (optionally) persist."""
    config.EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config.RANDOM_SEED)

    base = pd.read_excel(config.RAW_DATASET)
    base["Date"] = pd.to_datetime(base["Date"])
    base = base.rename(columns={"Target_Demand": "Base_Demand"})
    base = base.sort_values(["Region", "Date"]).reset_index(drop=True)

    geo = build_geography()
    weather = build_weather(base, geo, rng)
    climate = build_climate(base, geo, rng)
    political = build_political(base, geo, rng)
    festivals = build_festival_calendar(base)

    # Merge everything into one modelling table.
    df = base.merge(geo, on="Region", how="left")
    df = df.merge(weather, on=["Date", "Region"], how="left")
    df = df.merge(
        climate.drop(columns=["Climate_Zone"]), on=["Date", "Region"], how="left"
    )
    df = df.merge(political, on=["Date", "Region"], how="left")

    df = apply_demand_enrichment(df, rng)
    df = df.sort_values(["Region", "Date"]).reset_index(drop=True)

    if save:
        geo.to_csv(config.GEOGRAPHY_FILE, index=False)
        weather.to_csv(config.WEATHER_FILE, index=False)
        climate.to_csv(config.CLIMATE_FILE, index=False)
        political.to_csv(config.POLITICAL_FILE, index=False)
        festivals.to_csv(config.FESTIVAL_FILE, index=False)
        df.to_csv(config.ENRICHED_FILE, index=False)
        print(f"[data_generation] geography  -> {config.GEOGRAPHY_FILE.name}  ({len(geo)} rows)")
        print(f"[data_generation] weather    -> {config.WEATHER_FILE.name}  ({len(weather)} rows)")
        print(f"[data_generation] climate    -> {config.CLIMATE_FILE.name}  ({len(climate)} rows)")
        print(f"[data_generation] political  -> {config.POLITICAL_FILE.name}  ({len(political)} rows)")
        print(f"[data_generation] festivals  -> {config.FESTIVAL_FILE.name}  ({len(festivals)} rows)")
        print(f"[data_generation] ENRICHED   -> {config.ENRICHED_FILE.name}  "
              f"({len(df)} rows x {df.shape[1]} cols)")
        corr = np.corrcoef(df['Base_Demand'], df['Target_Demand'])[0, 1]
        print(f"[data_generation] corr(Base_Demand, Target_Demand) = {corr:.3f}")

    return df


if __name__ == "__main__":
    generate_all(save=True)
