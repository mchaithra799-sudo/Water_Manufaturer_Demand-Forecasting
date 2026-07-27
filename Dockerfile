# --- Water Bottle Demand Forecasting dashboard -----------------------------
FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal; scikit-learn wheels are self-contained.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the data + model + forecast artefacts at image-build time so the
# container starts instantly with a working dashboard.
RUN python run_pipeline.py

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').status==200 else 1)" || exit 1

CMD ["streamlit", "run", "app/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
