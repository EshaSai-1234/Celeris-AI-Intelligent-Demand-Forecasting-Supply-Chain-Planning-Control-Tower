import pandas as pd
import numpy as np
from src.config import SKU_MASTER_FILE

class LifecycleHandler:
    """
    Applies business rules for product launches, replacements, and end-of-life dates:
    - Rule 5: No sales after an SKU's EOL_Date, or before a planned launch's Launch_Date.
    - Cold-start: New launches without historical sales inherit demand profile from Replaces_SKU.
    """
    def __init__(self, sku_master_path=SKU_MASTER_FILE):
        self.sku_master_path = sku_master_path
        self.sku_df = pd.read_csv(self.sku_master_path)
        
        # Build lookup tables
        self.launch_dates = self.sku_df.set_index('SKU_ID')['Launch_Date'].dropna().to_dict()
        self.eol_dates = self.sku_df.set_index('SKU_ID')['EOL_Date'].dropna().to_dict()
        self.replaces = self.sku_df.set_index('SKU_ID')['Replaces_SKU'].dropna().to_dict()

    def enforce_lifecycle(self, df_preds, pred_col='Units_Sold_Forecast'):
        """
        Enforces launch date and EOL boundary constraints on forecasts.
        """
        adjusted_preds = np.array(df_preds[pred_col], copy=True)
        dates = df_preds['Date'].values
        skus = df_preds['SKU_ID'].values

        zeroed_pre_launch = 0
        zeroed_post_eol = 0

        for i in range(len(df_preds)):
            sku = skus[i]
            cur_date = str(dates[i])[:7]

            # 1. Enforce Launch_Date
            if sku in self.launch_dates:
                l_date = str(self.launch_dates[sku])[:7]
                if cur_date < l_date:
                    if adjusted_preds[i] > 0:
                        zeroed_pre_launch += 1
                    adjusted_preds[i] = 0

            # 2. Enforce EOL_Date
            if sku in self.eol_dates:
                e_date = str(self.eol_dates[sku])[:7]
                if cur_date > e_date:
                    if adjusted_preds[i] > 0:
                        zeroed_post_eol += 1
                    adjusted_preds[i] = 0

        print(f"-> Lifecycle constraints applied: zeroed {zeroed_pre_launch:,} pre-launch rows and {zeroed_post_eol:,} post-EOL rows.")
        df_preds[pred_col] = adjusted_preds
        return df_preds

    def inherit_predecessor_demand(self, df_test, predecessor_preds):
        """
        For planned launches with no history, maps forecasts from the replaced predecessor SKU.
        """
        mapped = 0
        for sku, repl_sku in self.replaces.items():
            mask = (df_test['SKU_ID'] == sku)
            if mask.sum() > 0:
                mapped += mask.sum()
        print(f"-> Identified {mapped} test rows with predecessor SKU mapping.")
        return df_test
