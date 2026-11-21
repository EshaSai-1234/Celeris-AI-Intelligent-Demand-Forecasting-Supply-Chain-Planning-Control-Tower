# Celeris AI — Demand Forecasting & Supply Chain Control Tower

Celeris AI is an end-to-end forecasting system that predicts unconstrained product demand and converts forecasts into actionable inventory and procurement decisions.

**Demo:** [Streamlit app demo](assets/streamlit-app-demo.webm)

<video src="assets/streamlit-app-demo.webm" controls muted width="100%"></video>

## Overview

- Cleans and validates 244,566 historical records and 58,219 forecast records
- Reconstructs unconstrained demand for stockouts and lost sales
- Detects anomalies and evaluates 64 engineered features
- Forecasts 19 months using LightGBM, XGBoost, seasonal baselines, and Croston-SBA
- Produces P10/P50/P90 intervals, SHAP explanations, inventory plans, and a validated submission
- Provides an interactive 10-page Streamlit control tower

## Evaluation

Chronological rolling-origin backtesting uses 3 six-month folds.

| Model | WAPE | MAE | RMSE | Forecast Bias |
|---|---:|---:|---:|---:|
| **LightGBM Global** | **33.56%** | **17.84** | **70.51** | **-1.26%** |
| Seasonal Naive | 33.96% | 18.18 | 67.92 | -7.39% |
| XGBoost Global | 43.74% | 24.36 | 88.07 | +17.76% |
| Croston-SBA | 50.61% | 27.38 | 75.04 | -12.30% |
| Moving Average (3M) | 51.30% | 27.64 | 82.43 | -9.22% |

The selected LightGBM model outperformed the baselines overall and was retrained for the final forecasts.

## Key Outputs

- `outputs/submission.csv` — 58,219 validated forecast records
- `outputs/reconciled_forecasts_detailed.csv` — reconciled P10/P50/P90 forecasts
- `outputs/inventory_plan.csv` — SKU-region procurement and inventory recommendations
- `outputs/anomaly_register.csv` — detected demand anomalies
- `outputs/model_comparison.csv` and `outputs/backtest_results.csv` — model performance
- `outputs/evaluation_metrics.txt` and `outputs/pipeline_run_summary.txt` — evaluation and pipeline text reports
- `models/` — production point and quantile LightGBM models
- `Project Dashboard Outputs/` — dashboard PDF exports

## Dashboard

The dashboard includes executive overview, data health, demand exploration, anomaly monitoring, forecast accuracy, macro analysis, scenario simulation, procurement planning, lifecycle monitoring, and stockout analysis.

Launch it with:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Run the Pipeline

```bash
python -m src
pytest -q
```

## Docker

```bash
docker build -t celeris-ai .
docker run --rm -p 8501:8501 celeris-ai
```

Open `http://localhost:8501`.
