import os
import pandas as pd
import numpy as np
from src.config import (
    CLEANED_TRAIN_FILE, CLEANED_TEST_FILE, SKU_MASTER_FILE,
    EVENTS_FILE, OUTPUTS_DIR
)

def build_intermittency_features(df_monthly):
    """
    Computes ADI and CV2 per SKU_ID and Region to classify demand pattern.
    """
    stats = []
    for (sku, reg), grp in df_monthly.groupby(['SKU_ID', 'Region']):
        demand = grp['Unconstrained_Demand'].values
        total_p = len(demand)
        non_zero = demand[demand > 0]
        n_nz = len(non_zero)
        if n_nz > 1:
            adi = total_p / n_nz
            mean_nz = np.mean(non_zero)
            std_nz = np.std(non_zero, ddof=1)
            cv2 = (std_nz / mean_nz) ** 2 if mean_nz > 0 else 0
        else:
            adi = 2.0
            cv2 = 1.0
        
        # Classification
        if adi < 1.32 and cv2 < 0.49:
            cat = 'Smooth'
        elif adi >= 1.32 and cv2 < 0.49:
            cat = 'Intermittent'
        elif adi < 1.32 and cv2 >= 0.49:
            cat = 'Erratic'
        else:
            cat = 'Lumpy'
            
        stats.append({
            'SKU_ID': sku,
            'Region': reg,
            'ADI': adi,
            'CV2': cv2,
            'Demand_Pattern': cat
        })
    return pd.DataFrame(stats)

class FeatureEngineer:
    def __init__(self, train_path=CLEANED_TRAIN_FILE, test_path=CLEANED_TEST_FILE,
                 sku_master_path=SKU_MASTER_FILE, events_path=EVENTS_FILE):
        self.train_path = train_path
        self.test_path = test_path
        self.sku_master_path = sku_master_path
        self.events_path = events_path
        self.macro_path = os.path.join(OUTPUTS_DIR, "cleaned_macro_indicators.csv")

    def create_features(self):
        print(">>> Starting Leakage-Safe Feature Engineering Pipeline...")
        df_train = pd.read_csv(self.train_path)
        df_test = pd.read_csv(self.test_path)
        df_sku = pd.read_csv(self.sku_master_path)
        df_events = pd.read_csv(self.events_path)
        df_macro = pd.read_csv(self.macro_path)

        # 1. Aggregate Training and Test at modeling grain:
        # Notice test.csv has grain: (Date, SKU_ID, Region, Customer_Name, Sales_Channel)
        # We model at this granular level so test rows map 1-to-1!
        print("-> Preparing base modeling frame...")
        
        # Mark split
        df_train['is_train'] = 1
        df_test['is_train'] = 0
        df_test['Units_Sold'] = np.nan
        df_test['Unconstrained_Demand'] = np.nan
        
        # Combine train & test for consistent feature generation
        cols_common = ['Date', 'Year', 'Month', 'Product_Family', 'Business_Unit', 
                       'SKU_ID', 'SKU_Description', 'Customer_Name', 'Sales_Channel', 
                       'Region', 'is_train', 'Units_Sold', 'Unconstrained_Demand']
        
        # Group train by the exact keys to ensure clean unique series
        df_train_agg = df_train.groupby(
            ['Date', 'Year', 'Month', 'Product_Family', 'Business_Unit', 'SKU_ID', 'SKU_Description', 'Customer_Name', 'Sales_Channel', 'Region'],
            as_index=False
        ).agg({
            'Units_Sold': 'sum',
            'Unconstrained_Demand': 'sum'
        })
        df_train_agg['is_train'] = 1

        combined = pd.concat([df_train_agg, df_test[cols_common]], ignore_index=True)
        combined = combined.sort_values(by=['SKU_ID', 'Region', 'Customer_Name', 'Sales_Channel', 'Date']).reset_index(drop=True)

        # 2. Calendar and Cyclical Features
        print("-> Generating calendar and cyclical features...")
        combined['Month_Sin'] = np.sin(2 * np.pi * combined['Month'] / 12.0)
        combined['Month_Cos'] = np.cos(2 * np.pi * combined['Month'] / 12.0)
        combined['Quarter'] = ((combined['Month'] - 1) // 3) + 1
        combined['Is_Q4'] = (combined['Quarter'] == 4).astype(int)
        combined['Is_Year_End'] = (combined['Month'] == 12).astype(int)
        combined['Is_Year_Start'] = (combined['Month'] == 1).astype(int)

        # 3. SKU Master Metadata Features
        print("-> Merging SKU Master lifecycle and application features...")
        sku_meta = df_sku[['SKU_ID', 'Size_Class', 'Powertrain_Application', 'Launch_Date', 'EOL_Date', 'Replaces_SKU', 'Lifecycle_Status']].copy()
        combined = combined.merge(sku_meta, on='SKU_ID', how='left')

        # Compute SKU Age in months
        def calc_sku_age(row):
            try:
                l_date = str(row['Launch_Date'])[:7]
                cur_date = str(row['Date'])[:7]
                y_l, m_l = int(l_date[:4]), int(l_date[5:7])
                y_c, m_c = int(cur_date[:4]), int(cur_date[5:7])
                return (y_c - y_l) * 12 + (m_c - m_l)
            except:
                return 36
        combined['SKU_Age_Months'] = combined.apply(calc_sku_age, axis=1)

        # 4. Macroeconomic Indicator Features & Lead/Lag
        print("-> Merging Macroeconomic indicators with lead/lag structure...")
        macro_cols = ['ISM_Manufacturing_PMI', 'Auto_Production_YoY_Pct', 'Vehicle_Inventory_Days',
                      'Raw_Material_Cost_Index', 'New_Vehicle_Orders_Index', 'Industrial_Util_Rate_Pct',
                      'Fuel_Price_Index', 'EV_Adoption_Index']
        df_macro_sub = df_macro[['Date'] + macro_cols].copy()
        
        # Add lag-1 and lag-2 macroeconomic indicators
        for col in macro_cols:
            df_macro_sub[f'{col}_lag1'] = df_macro_sub[col].shift(1)
            df_macro_sub[f'{col}_lag2'] = df_macro_sub[col].shift(2)
        
        df_macro_sub = df_macro_sub.bfill()
        combined = combined.merge(df_macro_sub, on='Date', how='left')

        # 5. Business Events Calendar Encoding
        print("-> Incorporating Business Events (price increases, lockdowns, holidays)...")
        # Global price increase event: January of every year
        combined['Event_Price_Increase'] = (combined['Month'] == 1).astype(int)
        # Pre-buying event: December ahead of price increase
        combined['Event_Pre_Price_Increase'] = (combined['Month'] == 12).astype(int)
        # COVID-19 disruption period: 2020-02 to 2020-06
        covid_dates = ['2020-02', '2020-03', '2020-04', '2020-05', '2020-06']
        combined['Event_Covid_Disruption'] = combined['Date'].isin(covid_dates).astype(int)
        # Semiconductor shortage period: 2021-03 to 2022-09
        chip_dates = [f"2021-{m:02d}" for m in range(3, 13)] + [f"2022-{m:02d}" for m in range(1, 10)]
        combined['Event_Chip_Shortage'] = combined['Date'].isin(chip_dates).astype(int)
        # European summer shutdown: August
        combined['Event_Summer_Shutdown'] = ((combined['Region'] == 'Europe') & (combined['Month'] == 8)).astype(int)
        # APAC Lunar New Year: January / February
        combined['Event_Lunar_New_Year'] = ((combined['Region'] == 'Asia Pacific') & (combined['Month'].isin([1, 2]))).astype(int)

        # 6. Intermittency Diagnostics (ADI & CV2)
        print("-> Calculating ADI and CV2 intermittency classification...")
        intermittency_df = build_intermittency_features(df_train)
        combined = combined.merge(intermittency_df[['SKU_ID', 'Region', 'ADI', 'CV2', 'Demand_Pattern']], on=['SKU_ID', 'Region'], how='left')
        combined['Demand_Pattern'] = combined['Demand_Pattern'].fillna('Smooth')
        combined['ADI'] = combined['ADI'].fillna(1.0)
        combined['CV2'] = combined['CV2'].fillna(0.3)

        # 7. SKU and Regional Historical Baseline Statistics (Leakage-safe from Train Only)
        print("-> Computing historical demand aggregations from train...")
        train_only = combined[combined['is_train'] == 1]
        
        # Continuous time index (trend)
        combined['Time_Index'] = (combined['Year'] - 2019) * 12 + combined['Month']
        
        # Granular Seasonal SKU-Region-Month baseline
        sku_reg_m = train_only.groupby(['SKU_ID', 'Region', 'Month'])['Unconstrained_Demand'].agg(
            feat_seasonal_baseline='mean'
        ).reset_index()
        combined = combined.merge(sku_reg_m, on=['SKU_ID', 'Region', 'Month'], how='left')

        # SKU-Month seasonal baseline
        sku_m = train_only.groupby(['SKU_ID', 'Month'])['Unconstrained_Demand'].agg(
            feat_sku_month_mean='mean'
        ).reset_index()
        combined = combined.merge(sku_m, on=['SKU_ID', 'Month'], how='left')

        # Family-Region-Month seasonal baseline
        fam_reg_m = train_only.groupby(['Product_Family', 'Region', 'Month'])['Unconstrained_Demand'].agg(
            feat_fam_reg_month_mean='mean'
        ).reset_index()
        combined = combined.merge(fam_reg_m, on=['Product_Family', 'Region', 'Month'], how='left')

        # SKU-Customer historical baseline
        sku_cust = train_only.groupby(['SKU_ID', 'Customer_Name'])['Unconstrained_Demand'].agg(
            feat_sku_cust_mean='mean'
        ).reset_index()
        combined = combined.merge(sku_cust, on=['SKU_ID', 'Customer_Name'], how='left')

        # SKU-Region historical stats
        sku_reg_stats = train_only.groupby(['SKU_ID', 'Region'])['Unconstrained_Demand'].agg(
            hist_sku_reg_mean='mean',
            hist_sku_reg_std='std',
            hist_sku_reg_median='median'
        ).reset_index()
        combined = combined.merge(sku_reg_stats, on=['SKU_ID', 'Region'], how='left')
        
        # SKU overall historical stats
        sku_stats = train_only.groupby('SKU_ID')['Unconstrained_Demand'].agg(
            hist_sku_mean='mean',
            hist_sku_std='std',
            hist_sku_median='median'
        ).reset_index()
        combined = combined.merge(sku_stats, on='SKU_ID', how='left')
        
        # Fill missing with hierarchical fallbacks
        combined['hist_sku_reg_mean'] = combined['hist_sku_reg_mean'].fillna(combined['hist_sku_mean']).fillna(20.0)
        combined['hist_sku_reg_std'] = combined['hist_sku_reg_std'].fillna(combined['hist_sku_std']).fillna(10.0)
        combined['hist_sku_reg_median'] = combined['hist_sku_reg_median'].fillna(combined['hist_sku_median']).fillna(15.0)

        combined['feat_seasonal_baseline'] = combined['feat_seasonal_baseline'].fillna(combined['feat_sku_month_mean']).fillna(combined['hist_sku_reg_mean'])
        combined['feat_sku_month_mean'] = combined['feat_sku_month_mean'].fillna(combined['hist_sku_mean']).fillna(20.0)
        combined['feat_fam_reg_month_mean'] = combined['feat_fam_reg_month_mean'].fillna(combined['hist_sku_reg_mean'])
        combined['feat_sku_cust_mean'] = combined['feat_sku_cust_mean'].fillna(combined['hist_sku_mean']).fillna(20.0)

        # Macro interactions
        combined['pmi_demand_multiplier'] = (combined['ISM_Manufacturing_PMI'] / 50.0) * combined['feat_seasonal_baseline']

        # Customer & Channel historical shares
        cust_stats = train_only.groupby('Customer_Name')['Unconstrained_Demand'].agg(hist_cust_mean='mean').reset_index()
        chan_stats = train_only.groupby('Sales_Channel')['Unconstrained_Demand'].agg(hist_chan_mean='mean').reset_index()
        combined = combined.merge(cust_stats, on='Customer_Name', how='left')
        combined = combined.merge(chan_stats, on='Sales_Channel', how='left')
        combined['hist_cust_mean'] = combined['hist_cust_mean'].fillna(combined['Unconstrained_Demand'].mean())
        combined['hist_chan_mean'] = combined['hist_chan_mean'].fillna(combined['Unconstrained_Demand'].mean())

        # 8. Encode Categorical Columns
        categorical_cols = [
            'SKU_ID', 'Product_Family', 'Business_Unit', 'Region', 'Customer_Name', 
            'Sales_Channel', 'Size_Class', 'Powertrain_Application', 'Demand_Pattern',
            'Lifecycle_Status'
        ]
        for col in categorical_cols:
            if col in combined.columns:
                combined[col] = combined[col].astype('category')

        # Split back into train and test
        feat_train = combined[combined['is_train'] == 1].reset_index(drop=True)
        feat_test = combined[combined['is_train'] == 0].reset_index(drop=True)

        print(f"-> Train features shape: {feat_train.shape}, Test features shape: {feat_test.shape}")
        print(">>> Feature Engineering Completed Successfully!\n")
        return feat_train, feat_test

if __name__ == '__main__':
    fe = FeatureEngineer()
    fe.create_features()
