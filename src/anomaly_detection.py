import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from src.config import CLEANED_TRAIN_FILE, ANOMALY_REGISTER_FILE, RANDOM_STATE

class AnomalyDetector:
    def __init__(self, train_path=CLEANED_TRAIN_FILE, output_path=ANOMALY_REGISTER_FILE):
        self.train_path = train_path
        self.output_path = output_path

    def run_detection(self, df_train=None):
        print(">>> Starting Anomaly Detection Pipeline...")
        if df_train is None:
            df_train = pd.read_csv(self.train_path)

        anomalies = []

        # 1. Flag Return / Adjustment Anomalies
        print("-> Identifying negative return adjustments...")
        returns_mask = df_train['Is_Return_Adjustment'] == 1
        for _, row in df_train[returns_mask].iterrows():
            anomalies.append({
                'Record_ID': row['Record_ID'],
                'Date': row['Date'],
                'SKU_ID': row['SKU_ID'],
                'Region': row['Region'],
                'Customer_Name': row['Customer_Name'],
                'Anomaly_Type': 'Return / Negative Adjustment',
                'Severity': 'High' if abs(row['Raw_Units_Sold']) > 50 else 'Medium',
                'Explanation': f"Negative shipment quantity ({row['Raw_Units_Sold']} units) representing a return or accounting credit.",
                'Recommended_Treatment': 'Clipped to zero for unconstrained demand forecasting; recorded in audit register.'
            })

        # 2. Flag Stockout Lost Sales Events
        print("-> Registering Stockout & Lost Sales Anomalies...")
        stockout_mask = df_train['Is_Stockout'] == 1
        for _, row in df_train[stockout_mask].iterrows():
            lost = row['Lost_Units']
            shipped = row['Units_Sold']
            fill_rate = row['Fill_Rate_Pct']
            reason = row['Stockout_Reason']
            severity = 'High' if fill_rate < 50 else ('Medium' if fill_rate < 80 else 'Low')
            anomalies.append({
                'Record_ID': row['Record_ID'],
                'Date': row['Date'],
                'SKU_ID': row['SKU_ID'],
                'Region': row['Region'],
                'Customer_Name': row['Customer_Name'],
                'Anomaly_Type': 'Supply Constraint / Stockout',
                'Severity': severity,
                'Explanation': f"Supply constrained due to '{reason}'. Fill rate {fill_rate:.1f}%, observed sales {shipped}, lost units ~{lost}.",
                'Recommended_Treatment': 'Reconstructed unconstrained customer demand using lost-sales uplift.'
            })

        # 3. Statistical Outlier Detection (Robust Rolling IQR on SKU-Region level)
        print("-> Performing Statistical Outlier Detection (IQR)...")
        # Group by SKU_ID and Region to calculate median and IQR of demand
        sku_stats = df_train.groupby(['SKU_ID', 'Region'])['Unconstrained_Demand'].agg(
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
            median='median',
            mean='mean'
        ).reset_index()
        sku_stats['iqr'] = sku_stats['q75'] - sku_stats['q25']
        sku_stats['upper_bound'] = sku_stats['q75'] + 2.5 * sku_stats['iqr']
        
        merged_stats = df_train.merge(sku_stats, on=['SKU_ID', 'Region'], how='left')
        iqr_spikes = merged_stats[
            (merged_stats['Unconstrained_Demand'] > merged_stats['upper_bound']) & 
            (merged_stats['Unconstrained_Demand'] > 30) &
            (merged_stats['Is_Stockout'] == 0) &
            (merged_stats['Is_Return_Adjustment'] == 0)
        ]
        
        for _, row in iqr_spikes.iterrows():
            # Check if December (early buying ahead of Jan 1 price increase)
            month = int(row['Month'])
            is_pre_buy = (month == 12)
            expl = "December pre-buying surge before January 1 price revision." if is_pre_buy else f"Demand surge: {row['Unconstrained_Demand']} units vs series median {row['median']:.1f} (IQR threshold {row['upper_bound']:.1f})."
            anomalies.append({
                'Record_ID': row['Record_ID'],
                'Date': row['Date'],
                'SKU_ID': row['SKU_ID'],
                'Region': row['Region'],
                'Customer_Name': row['Customer_Name'],
                'Anomaly_Type': 'Pre-Price Increase Surge' if is_pre_buy else 'Statistical Demand Surge',
                'Severity': 'Medium',
                'Explanation': expl,
                'Recommended_Treatment': 'Modeled using price-increase lead indicator and seasonal lag features; not removed.'
            })

        # 4. Machine Learning Multi-dimensional Outlier Detection (Isolation Forest)
        print("-> Running Multi-dimensional Isolation Forest...")
        feature_cols = ['Units_Sold', 'Unit_Price_USD', 'Revenue_USD']
        X = df_train[feature_cols].copy()
        iso = IsolationForest(n_estimators=100, contamination=0.005, random_state=RANDOM_STATE, n_jobs=-1)
        iso_preds = iso.fit_predict(X)
        iso_mask = (iso_preds == -1) & (df_train['Is_Return_Adjustment'] == 0) & (df_train['Is_Stockout'] == 0)
        
        for _, row in df_train[iso_mask].iterrows():
            anomalies.append({
                'Record_ID': row['Record_ID'],
                'Date': row['Date'],
                'SKU_ID': row['SKU_ID'],
                'Region': row['Region'],
                'Customer_Name': row['Customer_Name'],
                'Anomaly_Type': 'Multi-dimensional Outlier',
                'Severity': 'Medium',
                'Explanation': f"Isolation Forest flagged multidimensional anomaly (Units: {row['Units_Sold']}, Price: ${row['Unit_Price_USD']:.2f}, Rev: ${row['Revenue_USD']:.2f}).",
                'Recommended_Treatment': 'Retained in training set; tree models naturally handle non-linear multidimensional boundaries.'
            })

        # 5. Compile and Save Anomaly Register
        anomaly_df = pd.DataFrame(anomalies)
        # Deduplicate if a row was flagged under multiple categories
        anomaly_df = anomaly_df.drop_duplicates(subset=['Record_ID', 'Anomaly_Type']).reset_index(drop=True)
        anomaly_df.to_csv(self.output_path, index=False)
        print(f"-> Anomaly Register saved to {self.output_path} with {len(anomaly_df):,} detected records.")
        print("   Breakdown by Type:\n", anomaly_df['Anomaly_Type'].value_counts().to_string())
        print("   Breakdown by Severity:\n", anomaly_df['Severity'].value_counts().to_string())
        print(">>> Anomaly Detection Completed Successfully!\n")
        return anomaly_df

if __name__ == '__main__':
    detector = AnomalyDetector()
    detector.run_detection()
