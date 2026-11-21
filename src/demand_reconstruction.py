import os
import pandas as pd
import numpy as np
from src.config import CLEANED_TRAIN_FILE, STOCKOUTS_FILE

def parse_date_robust(val):
    if pd.isna(val):
        return val
    s = str(val).strip()
    return s[:7]

class DemandReconstructor:
    def __init__(self, train_path=CLEANED_TRAIN_FILE, stockouts_path=STOCKOUTS_FILE):
        self.train_path = train_path
        self.stockouts_path = stockouts_path

    def reconstruct(self, df_train=None):
        print(">>> Starting Unconstrained Demand Reconstruction...")
        if df_train is None:
            df_train = pd.read_csv(self.train_path)

        df_stockouts = pd.read_csv(self.stockouts_path)
        df_stockouts['Date'] = df_stockouts['Date'].apply(parse_date_robust)

        # Aggregate stockouts at (Date, SKU_ID, Region) in case multiple events occurred
        stockout_agg = df_stockouts.groupby(['Date', 'SKU_ID', 'Region']).agg({
            'Requested_Units': 'sum',
            'Shipped_Units': 'sum',
            'Backordered_Units': 'sum',
            'Lost_Units': 'sum',
            'Fill_Rate_Pct': 'mean',
            'Stockout_Days': 'sum',
            'Reason': lambda x: ' | '.join(x.unique())
        }).reset_index()

        print(f"   Aggregated stockout events: {len(stockout_agg)} SKU-Region-Month combinations.")

        # Left join with training data
        merged = df_train.merge(
            stockout_agg,
            on=['Date', 'SKU_ID', 'Region'],
            how='left'
        )

        # Flag stockout occurrence
        merged['Is_Stockout'] = merged['Lost_Units'].notna().astype(int)
        merged['Lost_Units'] = merged['Lost_Units'].fillna(0.0)
        merged['Stockout_Days'] = merged['Stockout_Days'].fillna(0.0)
        merged['Fill_Rate_Pct'] = merged['Fill_Rate_Pct'].fillna(100.0)
        merged['Stockout_Reason'] = merged['Reason'].fillna('None')
        merged = merged.drop(columns=['Reason'], errors='ignore')

        # Reconstruct Unconstrained Demand:
        # Base demand is observed Units_Sold.
        # If stockout occurred, Unconstrained Demand = Units_Sold + Lost_Units.
        # If Requested_Units is recorded and higher, use Requested_Units.
        has_stockout = merged['Is_Stockout'] == 1
        
        # When stockout has multiple customer transactions for that SKU-Region in a month,
        # distribute lost units proportionally or add to total
        # To avoid double-counting lost units across multiple customers for the same SKU-Region:
        # Calculate customer's share of shipments in that SKU-Region-Month
        region_sku_month_shipped = merged.groupby(['Date', 'SKU_ID', 'Region'])['Units_Sold'].transform('sum')
        cust_share = np.where(region_sku_month_shipped > 0, merged['Units_Sold'] / region_sku_month_shipped, 1.0)
        
        # Customer-allocated lost units
        cust_lost_units = np.round(merged['Lost_Units'] * cust_share)
        
        merged['Unconstrained_Demand'] = merged['Units_Sold'] + np.where(has_stockout, cust_lost_units, 0.0)
        
        # Ensure Unconstrained_Demand is non-negative and >= Units_Sold
        merged['Unconstrained_Demand'] = np.maximum(merged['Units_Sold'], merged['Unconstrained_Demand']).astype(int)

        # Calculate reconstruction uplift
        total_observed = merged['Units_Sold'].sum()
        total_unconstrained = merged['Unconstrained_Demand'].sum()
        total_lost = total_unconstrained - total_observed
        print(f"   Total Observed Units Sold:    {total_observed:,}")
        print(f"   Total Unconstrained Demand:   {total_unconstrained:,}")
        print(f"   Reconstructed Lost Demand:    {total_lost:,} units (+{(total_lost/total_observed)*100:.2f}%)")
        print(f"   Total Stockout Rows Impacted: {has_stockout.sum():,}")

        # Update cleaned_train.csv with reconstructed demand columns
        merged.to_csv(self.train_path, index=False)
        print(f"   Successfully updated {self.train_path} with Unconstrained_Demand.")
        print(">>> Unconstrained Demand Reconstruction Complete!\n")
        return merged

if __name__ == '__main__':
    reconstructor = DemandReconstructor()
    reconstructor.reconstruct()
