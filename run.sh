#!/usr/bin/env bash
# One-shot setup + launch for the Water Bottle Demand Forecasting dashboard.
#
#   ./run.sh            # install deps, build artefacts (if missing), launch app
#   ./run.sh --rebuild  # force-regenerate data + model before launching
#
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

echo "==> Installing dependencies"
$PYTHON -m pip install -r requirements.txt

if [[ "${1:-}" == "--rebuild" || ! -f models/demand_model.joblib ]]; then
  echo "==> Building data + model + forecasts"
  $PYTHON run_pipeline.py
fi

echo "==> Launching dashboard at http://localhost:8501"
exec streamlit run app/app.py
