# 💧 Water Bottle Demand Forecasting

An end-to-end machine-learning application that forecasts water-bottle
manufacturing demand for the **next week, month and year** — driven not just by
the historical sales series, but by four additional generated datasets covering
**weather, climate, geography and political / economic conditions**. It ships
with an **interactive Streamlit dashboard** you can run and deploy straight from
VS Code.

> Built on top of the provided `Water_Bottle_Demand_Dataset.xlsx` (3 years of
> daily demand across 4 zones) and `Demand_Forecasting_Pipeline.ipynb`.

---

## ✨ What's inside

| Capability | Where |
|---|---|
| Generate 4 enrichment datasets (weather / climate / geography / political) | `src/data_generation.py` |
| Feature engineering (temporal, cyclical, one-hot) | `src/features.py` |
| Auto-selecting model (Random Forest vs Gradient Boosting) + time-series CV | `src/train_model.py` |
| Scenario-based future forecast (week / month / year) with 95% bands | `src/forecast.py` |
| Interactive 6-page dashboard | `app/app.py` |
| One-command pipeline | `run_pipeline.py` |
| VS Code run/debug config, Docker, Streamlit Cloud | `.vscode/`, `Dockerfile` |

**Model accuracy (held-out most-recent year):** ~**93%** (MAPE ≈ 6.7%,
R² ≈ 0.86). Numbers are regenerated every run into `models/metrics.json`.

---

## 🚀 Quick start

### Option A — VS Code (recommended)

1. Open this folder in **VS Code** (`File ▸ Open Folder`). Accept the
   recommended extensions prompt (Python + Jupyter).
2. Open the **Command Palette** → `Tasks: Run Task`:
   - **1 · Install dependencies**
   - **2 · Run pipeline (data + train + forecast)**
   - **3 · Launch dashboard**
3. Or press **F5** and pick **“Streamlit: Run dashboard”** to run with the
   debugger attached.

The dashboard opens at **http://localhost:8501**.

### Option B — Terminal

```bash
pip install -r requirements.txt
python run_pipeline.py        # builds data/, models/  (~15s)
streamlit run app/app.py
```

### Option C — One script

```bash
./run.sh            # installs, builds if needed, launches
./run.sh --rebuild  # force-rebuild data + model first
```

> The repo already ships the generated `data/processed/*` and `models/*`, so the
> dashboard runs **out-of-the-box** even before you run the pipeline.

---

## 🗂️ Project structure

```
Water_Manufaturer_Demand-Forecasting/
├── app/
│   └── app.py                     # Interactive Streamlit dashboard (6 pages)
├── src/
│   ├── config.py                  # Paths, constants, forecast horizon
│   ├── data_generation.py         # Builds the 4 enrichment datasets + master table
│   ├── features.py                # Feature engineering / model matrix
│   ├── train_model.py             # Train, evaluate, select, persist the model
│   └── forecast.py                # Future scenarios → week/month/year forecast
├── data/
│   ├── raw/                       # Provided Water_Bottle_Demand_Dataset.xlsx
│   ├── external/                  # Generated: geography, weather, climate, political
│   └── processed/                 # enriched_demand.csv + forecast_* .csv
├── models/                        # demand_model.joblib, feature_columns.json, metrics.json
├── notebooks/
│   └── Demand_Forecasting_Pipeline.ipynb
├── .vscode/                       # launch.json, tasks.json, settings, extensions
├── .streamlit/config.toml         # Dashboard theme
├── run_pipeline.py                # data → train → forecast, one command
├── Dockerfile                     # Container deployment
├── run.sh                         # Setup + launch helper
└── requirements.txt
```

---

## 📊 The datasets

### Provided (backbone)
`data/raw/Water_Bottle_Demand_Dataset.xlsx` — 4,380 daily rows (2021–2023) ×
4 zones with `Temperature_C`, `Season`, `Is_Festival` and `Target_Demand`.

### Generated ("more data", as requested)
All four are built by `src/data_generation.py`, physically tied to the provided
series so every feature carries **real, learnable signal**:

| Dataset | File | Example fields |
|---|---|---|
| 🌍 **Geography** | `data/external/geography.csv` | latitude, longitude, served-market population, urbanisation, elevation, coastal, climate zone |
| 🌦️ **Weather** | `data/external/weather.csv` | humidity, rainfall, **heat index**, wind speed, UV index, extreme-weather flag |
| 🌡️ **Climate** | `data/external/climate.csv` | warming anomaly, **drought index**, air-quality index, **water-stress index** |
| 🏛️ **Political / Economic** | `data/external/political_economic.csv` | **plastic-bottle ban**, political stability, economic index, consumer confidence, tourism, elections, water tax |

Everything is merged into `data/processed/enriched_demand.csv`
(4,380 rows × 34 columns).

> **Transparency note.** The enrichment layer is *synthetic but principled*:
> weather is derived from the given temperature & season, geography is
> calibrated to each zone's demand baseline, and the modelling target
> `Target_Demand` is the provided demand adjusted by small, documented effects
> (e.g. a plastic-bottle ban shaves ~10% off single-use demand). The untouched
> original is preserved as `Base_Demand`, and the two stay **>0.97 correlated**,
> so the data remains anchored to what you supplied while the dashboard's
> "what-if" levers genuinely move the forecast. Swap `data_generation.py` for
> real API pulls (e.g. OpenWeather, World Bank) to productionise.

---

## 🤖 How the forecast works

1. **Enrich** — merge the 4 datasets into one daily table.
2. **Engineer features** — calendar + cyclical time encodings, one-hot zones /
   seasons / climate zones, plus every weather / climate / geo / policy driver.
3. **Train & select** — Random Forest and Histogram Gradient Boosting are
   evaluated on a **time-based hold-out** (train on 2021–2022, test on 2023) and
   with **5-fold rolling time-series CV**; the winner is refit on all history.
4. **Forecast the future** — for each zone and future day we build plausible
   driver values from **climatological normals** (seasonal averages) plus
   forward trends (warming, economic growth), carry policy state forward, and
   apply the recurring festival calendar. The model scores those rows to give a
   daily forecast, aggregated into **next-week / next-month / next-year** views
   with 95% prediction intervals.

Top demand drivers learned by the model: **temperature**, **served-market
population** (geography), **festivals**, **plastic-ban policy**, **humidity**,
**season**, **water-stress** & **drought** (climate), **tourism** & **elections**.

---

## 🖥️ The dashboard (6 pages)

- **📊 Overview** — KPIs, monthly history vs next-year forecast, demand share, top drivers.
- **📈 Historical Analysis** — filterable trends, seasonality, temperature-vs-demand, driver correlations.
- **🔮 Demand Forecast** — pick **Next Week / Month / Year** and a zone; forecast chart + table + CSV download.
- **🗺️ Geographical View** — zone map, market attributes, market-size vs demand.
- **🧪 What-If Simulator** — move weather / climate / economy / policy levers and watch predicted demand respond live.
- **🤖 Model & Data** — accuracy, validation strategy, feature importance, actual-vs-predicted, dataset downloads.

---

## ☁️ Deployment

### Streamlit Community Cloud (free)
1. Push this repo to GitHub.
2. Go to **share.streamlit.io** → *New app* → pick the repo/branch.
3. Set **Main file path** to `app/app.py`. Deploy. Done.

### Docker
```bash
docker build -t water-demand .
docker run -p 8501:8501 water-demand
# open http://localhost:8501
```

### Any host (Render / Railway / Azure / etc.)
Start command:
```bash
streamlit run app/app.py --server.port=$PORT --server.address=0.0.0.0
```

---

## 🔁 Regenerating everything

```bash
python run_pipeline.py     # or: python -m src.data_generation / .train_model / .forecast
```
Edit `src/config.py` to change the forecast horizon (`FORECAST_DAYS`) or
`src/data_generation.py` to plug in real weather/economic data sources.

---

## 🛠️ Tech stack

Python · pandas · scikit-learn · Streamlit · Plotly · statsmodels
