"""
Central configuration for the Water Bottle Demand Forecasting project.

All paths are resolved relative to the project root so the code works the same
whether you run it from VS Code, a terminal, or the Streamlit app.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EXTERNAL_DIR = DATA_DIR / "external"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_DATASET = RAW_DIR / "Water_Bottle_Demand_Dataset.xlsx"

# Generated external ("more") datasets requested by the brief
GEOGRAPHY_FILE = EXTERNAL_DIR / "geography.csv"
FESTIVAL_FILE = EXTERNAL_DIR / "festival_calendar.csv"
WEATHER_FILE = EXTERNAL_DIR / "weather.csv"
CLIMATE_FILE = EXTERNAL_DIR / "climate.csv"
POLITICAL_FILE = EXTERNAL_DIR / "political_economic.csv"

# Master modelling table (everything merged together)
ENRICHED_FILE = PROCESSED_DIR / "enriched_demand.csv"

# Forecast outputs
FORECAST_DAILY_FILE = PROCESSED_DIR / "forecast_daily.csv"
FORECAST_WEEKLY_FILE = PROCESSED_DIR / "forecast_weekly.csv"
FORECAST_MONTHLY_FILE = PROCESSED_DIR / "forecast_monthly.csv"
FORECAST_YEARLY_FILE = PROCESSED_DIR / "forecast_yearly.csv"

# Model artefacts
MODEL_FILE = MODELS_DIR / "demand_model.joblib"
FEATURE_COLUMNS_FILE = MODELS_DIR / "feature_columns.json"
METRICS_FILE = MODELS_DIR / "metrics.json"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_SEED = 42

# --------------------------------------------------------------------------- #
# Forecast horizon
# --------------------------------------------------------------------------- #
# The historical data ends on 2023-12-31, so "the next year" starts 2024-01-01.
# We generate a full 366-day (leap-year 2024) daily forecast and then aggregate
# it into next-week / next-month / next-year views inside the dashboard.
FORECAST_DAYS = 366

# --------------------------------------------------------------------------- #
# Domain constants
# --------------------------------------------------------------------------- #
REGIONS = ["North_Zone", "South_Zone", "East_Zone", "West_Zone"]
SEASONS = ["Winter", "Summer", "Monsoon", "Autumn"]

# Month -> Season mapping inferred from the provided dataset
MONTH_TO_SEASON = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Summer", 4: "Summer", 5: "Summer",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}

# The target column the model learns (enriched) and the untouched original.
TARGET_COL = "Target_Demand"
BASE_TARGET_COL = "Base_Demand"
