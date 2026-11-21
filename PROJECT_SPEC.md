# CELERIS AI – DEMAND FORECASTING & PLANNING CONTROL TOWER

## 1. Project Objective
Build an expert-level end-to-end AI/ML demand forecasting system for Celeris Automobili Industrial.

Predict `Units_Sold` for every row in `test.csv` from November 2026 through May 2028 (19 months).

Dataset scale:
- 600 SKUs
- 7 product families
- 6 global regions
- 248,248 training rows
- 58,219 test rows

## 2. Dataset Files
The `student_data/` directory contains:
- train.csv
- test.csv
- macro_indicators.csv
- sku_master.csv
- events_calendar.csv
- inventory_stockouts.csv
- sample_submission.csv

Inspect actual schemas before implementation. Never assume column names or relationships.

## 3. Business Rules
- Trust Date when Year/Month disagree.
- Resolve duplicate Record_ID using latest Load_Timestamp.
- Units_Sold = 0 means no order.
- Missing records are not automatically zero.
- Prices are SKU-based and generally change on January 1.
- Respect product launch and end-of-life dates.
- Estimate unconstrained demand, accounting for stockouts and lost sales.
- Distinguish data errors from genuine business events.
- Prevent future data leakage.

## 4. Data Engineering
Implement:
- Schema and data type validation.
- Duplicate detection and resolution.
- Missing-value analysis and treatment.
- Date and categorical normalization.
- Revenue consistency checks.
- SKU and region validation.
- Data-quality reporting.
- Reproducible preprocessing pipeline.

Outputs:
- cleaned_train.csv
- cleaned_test.csv
- data_quality_report.xlsx

## 5. Anomaly Detection
Use:
- IQR and robust statistical detection.
- Isolation Forest.
- Demand change and structural break detection.
- Stockout identification.
- Customer ordering-pattern analysis.

Create an anomaly register containing record identifier, anomaly type, severity, explanation and recommended treatment.

Do not automatically remove legitimate demand spikes or business disruptions.

## 6. Demand Reconstruction
Estimate unconstrained demand using historical sales, stockout information and lost-sales indicators.

Keep actual observed sales and reconstructed demand separately identifiable.

Document assumptions and confidence levels. Do not fabricate missing lost-sales quantities.

## 7. Feature Engineering
Create leakage-safe features:
- Calendar and seasonal features.
- Lag features: 1, 2, 3, 6 and 12 months.
- Rolling means and standard deviations.
- SKU age and lifecycle indicators.
- Product family, region, channel and customer features.
- Macroeconomic indicators.
- Lead/lag economic features using only information available at prediction time.
- EV adoption, fuel prices and vehicle-order indicators.
- Business event features.

## 8. Forecasting Models

Baselines:
- Seasonal Naive.
- Moving Average.

Advanced:
- Global LightGBM or XGBoost.
- SARIMA/Prophet where appropriate.
- Croston/SBA for intermittent demand.

Expert:
- Quantile LightGBM for P10/P50/P90.
- New-product launch handling.
- End-of-life demand handling.
- Hierarchical reconciliation using bottom-up or MinT.

Select models through actual validation performance, not assumptions.

## 9. Model Evaluation
Use chronological rolling-origin validation.

Evaluate multiple forecast horizons, including 19-month horizons where historical data permits.

Metrics:
- MAE
- RMSE
- WAPE
- Forecast Bias
- Quantile/interval coverage

Compare all advanced models against baselines.

Avoid random train/test splitting and leakage.

## 10. Explainable AI
Implement SHAP feature importance and SKU-level explanations.

Explain major demand increases, decreases and unusual forecasts.

## 11. Inventory Optimization
Implement:
- Safety stock.
- Reorder points.
- Inventory risk classification.
- Shortage and overstock risk.
- What-if demand scenarios.
- Procurement recommendations.

Clearly distinguish calculated metrics from estimates requiring lead-time or service-level assumptions.

## 12. Streamlit Dashboard

Mandatory pages:
1. Executive Overview.
2. Data Health.
3. Demand Explorer.
4. Anomaly Monitor.
5. Forecast & Accuracy.

Optional expert pages:
6. Macro Analysis.
7. Scenario Simulator.
8. Procurement Planner.
9. Product Lifecycle Monitor.
10. Stockout Analysis.

Shared filters:
- Family.
- SKU.
- Region.
- Channel.
- Customer.

Use Plotly for interactive charts.

## 13. Technical Stack
- Python
- Pandas and NumPy
- Scikit-learn
- LightGBM / XGBoost
- Statsmodels
- SHAP
- SciPy
- Streamlit
- Plotly
- Pytest
- Joblib

Use only necessary dependencies.

## 14. Final Deliverables
- Reproducible source code.
- Cleaned datasets.
- Data-quality report.
- Anomaly register.
- Feature engineering pipeline.
- Trained forecasting models.
- Backtesting and model comparison reports.
- P10/P50/P90 forecasts.
- SHAP explanations.
- Inventory recommendations.
- Interactive dashboard.
- Final submission.csv.
- README.md.
- requirements.txt.
- Dockerfile.

## 15. Submission Requirements
- Match sample_submission.csv schema exactly.
- Preserve test.csv row order and identifiers.
- Generate one valid prediction per test record.
- No missing, infinite or invalid predictions.
- Apply nonnegative demand constraints where appropriate.
- Validate the final CSV before saving.

## 16. Development Instructions
1. Inspect all seven CSV files first.
2. Generate a dataset profiling report.
3. Map actual columns and relationships.
4. Create an implementation plan.
5. Build modular, reusable components.
6. Implement and test incrementally.
7. Validate every major stage.
8. Run the complete pipeline.
9. Generate the final submission.
10. Document assumptions, limitations and actual results.

Never fabricate data, columns, metrics, validation results or business conclusions.

Prioritize official hackathon requirements, reproducibility, forecasting accuracy and business usefulness.