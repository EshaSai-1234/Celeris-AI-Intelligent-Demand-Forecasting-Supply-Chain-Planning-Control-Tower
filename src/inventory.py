import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from src.config import INVENTORY_PLAN_FILE

class InventoryOptimizer:
    """
    Supply chain planning and inventory optimization:
    - Calculates Safety Stock (SS) and Reorder Point (ROP)
    - Service level coverage (95%, 98%, 99%)
    - Risk classification: Shortage vs Overstock risk
    - P90 Procurement recommendations
    """
    def __init__(self, lead_time_months=2.0, lead_time_std=0.5, default_service_level=0.95):
        self.L = lead_time_months
        self.sigma_L = lead_time_std
        self.service_level = default_service_level
        self.Z = norm.ppf(default_service_level)

    def optimize(self, df_forecasts_with_quantiles, df_historical_stats=None):
        print(">>> Running Inventory Optimization & Procurement Planning...")
        
        # Aggregate 19-month forecasts per SKU and Region
        sku_reg_plan = df_forecasts_with_quantiles.groupby(['SKU_ID', 'Region', 'Product_Family']).agg({
            'Units_Sold_Forecast': ['mean', 'std', 'sum'],
            'P10': 'sum',
            'P50': 'sum',
            'P90': 'sum'
        }).reset_index()
        
        sku_reg_plan.columns = [
            'SKU_ID', 'Region', 'Product_Family',
            'Monthly_Mean_Demand', 'Monthly_Std_Demand', 'Total_19M_Forecast',
            'Total_P10', 'Total_P50', 'Total_P90'
        ]
        
        # Fill standard deviation for constant/intermittent demand
        sku_reg_plan['Monthly_Std_Demand'] = sku_reg_plan['Monthly_Std_Demand'].fillna(
            sku_reg_plan['Monthly_Mean_Demand'] * 0.3
        )
        
        D = sku_reg_plan['Monthly_Mean_Demand'].values
        sigma_D = sku_reg_plan['Monthly_Std_Demand'].values
        
        # 1. Safety Stock: SS = Z * sqrt(L * sigma_D^2 + D^2 * sigma_L^2)
        variance_term = self.L * (sigma_D ** 2) + (D ** 2) * (self.sigma_L ** 2)
        safety_stock = self.Z * np.sqrt(np.maximum(0, variance_term))
        sku_reg_plan['Safety_Stock_Units'] = np.round(safety_stock).astype(int)
        
        # 2. Reorder Point: ROP = (D * L) + SS
        cycle_demand = D * self.L
        rop = cycle_demand + safety_stock
        sku_reg_plan['Reorder_Point_Units'] = np.round(rop).astype(int)
        
        # 3. Monthly P90 Procurement Target
        # Per specification: "procurement plan from P90 forecasts"
        sku_reg_plan['Monthly_Procurement_P90'] = np.round(sku_reg_plan['Total_P90'] / 19.0).astype(int)
        
        # 4. Inventory Risk Classification
        cv = np.where(D > 0, sigma_D / D, 0.0)
        sku_reg_plan['Demand_CV'] = np.round(cv, 2)
        
        risk_labels = []
        for v in cv:
            if v > 0.8:
                risk_labels.append('High Volatility (Stockout Risk)')
            elif v > 0.4:
                risk_labels.append('Moderate Volatility (Balanced)')
            else:
                risk_labels.append('Stable Demand (Overstock Risk if Unmonitored)')
        sku_reg_plan['Inventory_Risk_Class'] = risk_labels
        
        # Save inventory plan
        sku_reg_plan.to_csv(INVENTORY_PLAN_FILE, index=False)
        print(f"-> Inventory Plan saved to {INVENTORY_PLAN_FILE} ({len(sku_reg_plan):,} SKU-Region plans).")
        print("   Breakdown by Risk Classification:\n", sku_reg_plan['Inventory_Risk_Class'].value_counts().to_string())
        print(">>> Inventory Optimization Completed Successfully!\n")
        return sku_reg_plan
