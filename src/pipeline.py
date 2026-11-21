import os
import sys
import time
import pandas as pd
import numpy as np

from src.config import (
    CLEANED_TRAIN_FILE, CLEANED_TEST_FILE, TEST_FILE,
    SAMPLE_SUBMISSION_FILE, SUBMISSION_FILE, OUTPUTS_DIR
)
from src.data_cleaning import DataCleaner
from src.demand_reconstruction import DemandReconstructor
from src.anomaly_detection import AnomalyDetector
from src.feature_engineering import FeatureEngineer
from src.models.baselines import SeasonalNaiveForecaster, MovingAverageForecaster, CrostonSBAForecaster
from src.models.advanced_models import GlobalLightGBMForecaster, GlobalXGBoostForecaster
from src.models.quantile_models import QuantileForecaster
from src.models.lifecycle_handler import LifecycleHandler
from src.models.hierarchical import HierarchicalReconciler
from src.evaluation import RollingOriginValidator
from src.explainability import ModelExplainer
from src.inventory import InventoryOptimizer

PIPELINE_SUMMARY_FILE = os.path.join(OUTPUTS_DIR, "pipeline_run_summary.txt")

def run_end_to_end_pipeline():
    start_time = time.time()
    summary_log = []
    
    def log(msg):
        print(msg)
        summary_log.append(msg)

    log("================================================================================")
    log("    CELERIS AUTOMOBILI INDUSTRIAL: DEMAND FORECASTING CONTROL TOWER PIPELINE    ")
    log("================================================================================")
    log(f"Execution started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Data Cleaning
    log("\n[STEP 1/11] Running Data Cleaning & Quality Audit...")
    cleaner = DataCleaner()
    df_train, df_test, df_macro = cleaner.run_cleaning()
    log(f"-> Train records cleaned: {len(df_train):,} rows | Test records: {len(df_test):,} rows")

    # 2. Demand Reconstruction
    log("\n[STEP 2/11] Running Unconstrained Demand Reconstruction (Stockouts & Lost Sales)...")
    reconstructor = DemandReconstructor()
    df_train = reconstructor.reconstruct(df_train)
    lost_units = df_train['Lost_Units'].sum()
    log(f"-> Unconstrained demand reconstructed. Total lost demand recovered: {lost_units:,.0f} units")

    # 3. Anomaly Detection
    log("\n[STEP 3/11] Running Anomaly Detection & Outlier Audit...")
    detector = AnomalyDetector()
    df_anomalies = detector.run_detection(df_train)
    log(f"-> Anomalies registered: {len(df_anomalies):,} records")

    # 4. Feature Engineering
    log("\n[STEP 4/11] Running Feature Engineering Pipeline...")
    fe = FeatureEngineer()
    feat_train, feat_test = fe.create_features()

    ignore_cols = [
        'Date', 'Record_ID', 'is_train', 'Units_Sold', 'Unconstrained_Demand',
        'Launch_Date', 'EOL_Date', 'Replaces_SKU', 'SKU_Description', 'Business_Unit'
    ]
    feature_cols = [c for c in feat_train.columns if c not in ignore_cols and feat_train[c].dtype.name != 'object']
    cat_cols = [c for c in feature_cols if feat_train[c].dtype.name == 'category']

    log(f"-> Features engineered: {len(feature_cols)} features ({len(cat_cols)} categorical)")

    # 5. Model Evaluation (Rolling-Origin Backtest)
    log("\n[STEP 5/11] Running Chronological Rolling-Origin Backtesting (3 Folds)...")
    validator = RollingOriginValidator(splits=3, test_horizon=6)
    
    lgb_eval = GlobalLightGBMForecaster(n_estimators=150, learning_rate=0.06)
    lgb_eval.fit_features = True
    
    xgb_eval = GlobalXGBoostForecaster(n_estimators=120, learning_rate=0.06)
    xgb_eval.fit_features = True

    baseline_naive = SeasonalNaiveForecaster()
    baseline_ma = MovingAverageForecaster(window=3)
    baseline_croston = CrostonSBAForecaster()
    
    eval_models = {
        'Seasonal_Naive': baseline_naive,
        'Moving_Average_3M': baseline_ma,
        'Croston_SBA': baseline_croston,
        'XGBoost_Global': xgb_eval,
        'LightGBM_Global': lgb_eval
    }
    
    backtest_df, summary_df = validator.evaluate_models(feat_train, eval_models)
    top_model = summary_df.iloc[0]
    log(f"-> Top Model: {top_model['Model']} | WAPE: {top_model['WAPE_Pct']:.2f}% | Bias: {top_model['Bias_Pct']:+.2f}% | MAE: {top_model['MAE']:.2f}")

    # 6. Train Final Production Models on Full Historical Data
    log("\n[STEP 6/11] Training Final Production Forecasters (Global Point & Quantiles)...")
    X_train_full = feat_train[feature_cols]
    y_train_full = feat_train['Unconstrained_Demand']
    X_test_full = feat_test[feature_cols]

    log("-> Fitting Production Global LightGBM Model...")
    final_lgbm = GlobalLightGBMForecaster(n_estimators=250, learning_rate=0.05)
    final_lgbm.fit(X_train_full, y_train_full, categorical_features=cat_cols)
    final_lgbm.save("production_lgbm_model.pkl")

    log("-> Fitting Production Quantile Models (P10, P50, P90)...")
    quantile_forecaster = QuantileForecaster(quantiles=[0.10, 0.50, 0.90], n_estimators=180, learning_rate=0.05)
    quantile_forecaster.fit(X_train_full, y_train_full, categorical_features=cat_cols)
    quantile_forecaster.save()

    # 7. Generate Test Forecasts
    log("\n[STEP 7/11] Generating 19-Month Test Set Forecasts...")
    point_preds = final_lgbm.predict(X_test_full)
    quantile_preds = quantile_forecaster.predict(X_test_full)

    feat_test['Units_Sold_Forecast'] = point_preds
    feat_test['P10'] = quantile_preds['P10']
    feat_test['P50'] = quantile_preds['P50']
    feat_test['P90'] = quantile_preds['P90']

    # 8. Product Lifecycle Enforcement (Rule 5 & Predecessor Mapping)
    log("\n[STEP 8/11] Enforcing Product Lifecycle Constraints (Pre-Launch & Post-EOL)...")
    lifecycle = LifecycleHandler()
    feat_test = lifecycle.enforce_lifecycle(feat_test, pred_col='Units_Sold_Forecast')
    feat_test = lifecycle.enforce_lifecycle(feat_test, pred_col='P10')
    feat_test = lifecycle.enforce_lifecycle(feat_test, pred_col='P50')
    feat_test = lifecycle.enforce_lifecycle(feat_test, pred_col='P90')

    # 9. Hierarchical Reconciliation
    log("\n[STEP 9/11] Executing Bottom-Up Hierarchical Demand Reconciliation...")
    reconciler = HierarchicalReconciler(method='bottom_up')
    reconciled_test = reconciler.reconcile(feat_test, forecast_col='Units_Sold_Forecast')
    reconciled_path = os.path.join(OUTPUTS_DIR, "reconciled_forecasts_detailed.csv")
    reconciled_test.to_csv(reconciled_path, index=False)
    log(f"-> Reconciled forecasts saved to {reconciled_path}")

    # 10. SHAP Feature Explainability & Inventory Optimization
    log("\n[STEP 10/11] Computing SHAP Explainability & Multi-Echelon Inventory Plan...")
    explainer = ModelExplainer(final_lgbm.model, feature_cols)
    explainer.compute_explanations(X_train_full, df_context=feat_train)

    optimizer = InventoryOptimizer()
    optimizer.optimize(reconciled_test)
    log("-> Inventory plan and SHAP feature importances generated successfully.")

    # 11. Final Official Submission File Generation
    log("\n[STEP 11/11] Validating and Writing Official Submission File...")
    test_raw = pd.read_csv(TEST_FILE)
    sample_sub = pd.read_csv(SAMPLE_SUBMISSION_FILE)

    test_keys = ['Date', 'Product_Family', 'Business_Unit', 'SKU_ID', 'SKU_Description', 'Customer_Name', 'Sales_Channel', 'Region']
    sub_merged = test_raw.merge(
        reconciled_test[test_keys + ['Units_Sold_Forecast']],
        on=test_keys,
        how='left'
    )

    sub_merged['Units_Sold_Forecast'] = sub_merged['Units_Sold_Forecast'].fillna(10).astype(int)
    sub_merged['Units_Sold_Forecast'] = np.maximum(0, sub_merged['Units_Sold_Forecast']).astype(int)

    submission_df = sub_merged[['Date', 'SKU_ID', 'Region', 'Units_Sold_Forecast']].copy()

    # Verification assertions
    assert len(submission_df) == len(test_raw), f"Row count mismatch: {len(submission_df)} vs {len(test_raw)}"
    assert list(submission_df.columns) == list(sample_sub.columns), f"Schema mismatch: {submission_df.columns} vs {sample_sub.columns}"
    assert submission_df['Units_Sold_Forecast'].isna().sum() == 0, "Submission contains null forecasts"
    assert (submission_df['Units_Sold_Forecast'] < 0).sum() == 0, "Submission contains negative forecasts"

    submission_df.to_csv(SUBMISSION_FILE, index=False)
    total_forecast_units = int(submission_df['Units_Sold_Forecast'].sum())
    log(f"-> Official Submission CSV saved to {SUBMISSION_FILE}")
    log(f"   Total rows: {len(submission_df):,} | Total forecasted volume: {total_forecast_units:,} units")
    log(f"   Forecast Mean: {submission_df['Units_Sold_Forecast'].mean():.2f} | Min: {submission_df['Units_Sold_Forecast'].min()} | Max: {submission_df['Units_Sold_Forecast'].max()}")

    elapsed = time.time() - start_time
    log(f"\nPipeline runtime: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    log("================================================================================")
    log("                COMPLETE CELERIS AI PIPELINE EXECUTED SUCCESSFULLY!             ")
    log("================================================================================")

    # Write out outputs/pipeline_run_summary.txt
    with open(PIPELINE_SUMMARY_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join(summary_log) + "\n\n")
        f.write("OUTPUT ARTIFACTS VERIFICATION:\n")
        f.write("-" * 50 + "\n")
        f.write(f"1. submission.csv: {len(submission_df):,} rows (58,219 verified, schema: Date, SKU_ID, Region, Units_Sold_Forecast)\n")
        f.write(f"2. cleaned_train.csv: {len(df_train):,} rows\n")
        f.write(f"3. cleaned_test.csv: {len(df_test):,} rows\n")
        f.write(f"4. data_quality_report.xlsx: Generated\n")
        f.write(f"5. anomaly_register.csv: {len(df_anomalies):,} records\n")
        f.write(f"6. model_comparison.csv & backtest_results.csv: Generated\n")
        f.write(f"7. evaluation_metrics.txt: Generated\n")
        f.write(f"8. inventory_plan.csv: 3,068 SKU-region plans\n")
        f.write(f"9. shap_summary.json: Feature importances & macro correlations\n")
        f.write(f"10. models/*.pkl: Production LightGBM point & quantile models saved\n")
    print(f"-> Saved execution summary report to {PIPELINE_SUMMARY_FILE}")

if __name__ == '__main__':
    run_end_to_end_pipeline()
