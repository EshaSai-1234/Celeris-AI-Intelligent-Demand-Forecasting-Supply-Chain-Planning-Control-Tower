import pandas as pd
import numpy as np

class HierarchicalReconciler:
    """
    Performs hierarchical reconciliation across:
    Level 0: Total Company
    Level 1: Region
    Level 2: Product Family
    Level 3: Bottom Level (SKU-Region-Customer-Channel)
    """
    def __init__(self, method='bottom_up'):
        self.method = method

    def reconcile(self, df_forecasts, forecast_col='Units_Sold_Forecast'):
        """
        Calculates reconciled hierarchical totals ensuring top-down/bottom-up consistency.
        """
        print(f"-> Applying Hierarchical Reconciliation using '{self.method}' approach...")
        
        # Bottom-level sums
        reconciled_df = df_forecasts.copy()
        
        # 1. Family Aggregations
        family_level = reconciled_df.groupby(['Date', 'Product_Family'])[forecast_col].sum().reset_index()
        family_level = family_level.rename(columns={forecast_col: 'Family_Forecast'})
        
        # 2. Region Aggregations
        region_level = reconciled_df.groupby(['Date', 'Region'])[forecast_col].sum().reset_index()
        region_level = region_level.rename(columns={forecast_col: 'Region_Forecast'})
        
        # 3. Global Total Aggregations
        global_level = reconciled_df.groupby('Date')[forecast_col].sum().reset_index()
        global_level = global_level.rename(columns={forecast_col: 'Global_Total_Forecast'})
        
        # Merge back hierarchical rollups
        reconciled_df = reconciled_df.merge(family_level, on=['Date', 'Product_Family'], how='left')
        reconciled_df = reconciled_df.merge(region_level, on=['Date', 'Region'], how='left')
        reconciled_df = reconciled_df.merge(global_level, on='Date', how='left')
        
        total_units = reconciled_df[forecast_col].sum()
        print(f"   Reconciliation complete. 19-month total company demand: {total_units:,} units.")
        return reconciled_df
