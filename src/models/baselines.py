import numpy as np
import pandas as pd

class SeasonalNaiveForecaster:
    """
    Seasonal Naive forecaster: uses same month from 12 months ago if available,
    otherwise falls back to the SKU-region overall mean.
    """
    def __init__(self):
        self.sku_month_means = {}
        self.sku_overall_means = {}

    def fit(self, df_train):
        # Compute mean demand per (SKU_ID, Region, Month)
        self.sku_month_means = df_train.groupby(
            ['SKU_ID', 'Region', 'Month']
        )['Unconstrained_Demand'].mean().to_dict()
        
        # Compute overall mean per (SKU_ID, Region)
        self.sku_overall_means = df_train.groupby(
            ['SKU_ID', 'Region']
        )['Unconstrained_Demand'].mean().to_dict()
        
        self.global_mean = df_train['Unconstrained_Demand'].mean()
        return self

    def predict(self, df_test):
        preds = []
        for _, row in df_test.iterrows():
            key_m = (row['SKU_ID'], row['Region'], int(row['Month']))
            key_all = (row['SKU_ID'], row['Region'])
            val = self.sku_month_means.get(key_m, self.sku_overall_means.get(key_all, self.global_mean))
            preds.append(max(0, round(val)))
        return np.array(preds)


class MovingAverageForecaster:
    """
    Trailing Moving Average Forecaster over window k (e.g., 3 or 6 months).
    """
    def __init__(self, window=3):
        self.window = window
        self.sku_reg_ma = {}
        self.global_mean = 0

    def fit(self, df_train):
        self.global_mean = df_train['Unconstrained_Demand'].mean()
        # Sort chronologically
        sorted_df = df_train.sort_values(by=['SKU_ID', 'Region', 'Date'])
        # Compute the last window observations per SKU-Region
        for (sku, reg), grp in sorted_df.groupby(['SKU_ID', 'Region']):
            tail_vals = grp['Unconstrained_Demand'].tail(self.window).values
            if len(tail_vals) > 0:
                self.sku_reg_ma[(sku, reg)] = np.mean(tail_vals)
            else:
                self.sku_reg_ma[(sku, reg)] = self.global_mean
        return self

    def predict(self, df_test):
        preds = []
        for _, row in df_test.iterrows():
            key = (row['SKU_ID'], row['Region'])
            val = self.sku_reg_ma.get(key, self.global_mean)
            preds.append(max(0, round(val)))
        return np.array(preds)


class CrostonSBAForecaster:
    """
    Croston's Method with Syntetos-Boylan Approximation (SBA) for intermittent demand.
    Applies SBA bias-correction factor (1 - alpha/2).
    """
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.forecasts = {}
        self.global_mean = 0

    def fit(self, df_train):
        self.global_mean = df_train['Unconstrained_Demand'].mean()
        sorted_df = df_train.sort_values(by=['SKU_ID', 'Region', 'Date'])
        
        for (sku, reg), grp in sorted_df.groupby(['SKU_ID', 'Region']):
            y = grp['Unconstrained_Demand'].values
            if len(y) == 0:
                self.forecasts[(sku, reg)] = self.global_mean
                continue
                
            non_zeros = np.where(y > 0)[0]
            if len(non_zeros) == 0:
                self.forecasts[(sku, reg)] = 0.0
                continue
                
            z = y[non_zeros[0]]  # Initial demand level
            p = 1.0              # Initial inter-arrival period
            q = 1                # Counter
            
            for t in range(1, len(y)):
                if y[t] > 0:
                    z = z + self.alpha * (y[t] - z)
                    p = p + self.alpha * (q - p)
                    q = 1
                else:
                    q += 1
                    
            # SBA correction: (1 - alpha/2) * (z / p)
            sba_forecast = (1.0 - self.alpha / 2.0) * (z / max(p, 0.001))
            self.forecasts[(sku, reg)] = max(0.0, sba_forecast)
            
        return self

    def predict(self, df_test):
        preds = []
        for _, row in df_test.iterrows():
            key = (row['SKU_ID'], row['Region'])
            val = self.forecasts.get(key, self.global_mean)
            preds.append(max(0, round(val)))
        return np.array(preds)
