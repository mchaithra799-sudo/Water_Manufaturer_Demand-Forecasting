"""
One-command pipeline: generate data -> train model -> forecast.

    python run_pipeline.py

Running this regenerates every artefact the dashboard needs
(data/processed/*, models/*).  It is safe to re-run at any time.
"""
from __future__ import annotations

import time

from src import data_generation, train_model, forecast


def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("WATER BOTTLE DEMAND FORECASTING - FULL PIPELINE")
    print("=" * 70)

    print("\n[1/3] Generating enriched dataset "
          "(geography + weather + climate + political) ...")
    data_generation.generate_all(save=True)

    print("\n[2/3] Training & evaluating the forecasting model ...")
    train_model.train(save=True)

    print("\n[3/3] Forecasting next-year / month / week demand ...")
    forecast.run_forecast(save=True)

    print("\n" + "=" * 70)
    print(f"DONE in {time.time() - t0:.1f}s. Launch the dashboard with:")
    print("    streamlit run app/app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
