import os
import re
import pandas as pd
import numpy as np
from datetime import datetime
from src.config import (
    TRAIN_FILE, TEST_FILE, MACRO_FILE, SKU_MASTER_FILE,
    CLEANED_TRAIN_FILE, CLEANED_TEST_FILE, DATA_QUALITY_REPORT_FILE
)

MONTH_MAP = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
    'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
    'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12
}

def parse_date_robust(val):
    """
    Parses messy date strings into canonical YYYY-MM.
    Handles 'YYYY-MM', 'YYYY/MM', 'MM/YYYY', 'YYYY-MM-DD', 'Jan 2019', 'Feb-2022', etc.
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    
    # Check YYYY-MM or YYYY/MM
    m1 = re.match(r"^(\d{4})[-/](\d{1,2})$", s)
    if m1:
        y, m = int(m1.group(1)), int(m1.group(2))
        return f"{y:04d}-{m:02d}"
        
    # Check MM/YYYY or MM-YYYY
    m2 = re.match(r"^(\d{1,2})[-/](\d{4})$", s)
    if m2:
        m, y = int(m2.group(1)), int(m2.group(2))
        return f"{y:04d}-{m:02d}"
        
    # Check YYYY-MM-DD
    m3 = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})", s)
    if m3:
        y, m = int(m3.group(1)), int(m3.group(2))
        return f"{y:04d}-{m:02d}"

    # Check 'Jan 2019', 'Feb-2022', 'March 2020'
    m4 = re.match(r"^([a-zA-Z]+)[-\s/]+(\d{4})$", s)
    if m4:
        mon_str = m4.group(1).lower()
        y = int(m4.group(2))
        if mon_str in MONTH_MAP:
            return f"{y:04d}-{MONTH_MAP[mon_str]:02d}"
            
    # Check '2019 Jan', '2022-Feb'
    m5 = re.match(r"^(\d{4})[-\s/]+([a-zA-Z]+)$", s)
    if m5:
        y = int(m5.group(1))
        mon_str = m5.group(2).lower()
        if mon_str in MONTH_MAP:
            return f"{y:04d}-{MONTH_MAP[mon_str]:02d}"

    # Fallback to pandas to_datetime
    try:
        dt = pd.to_datetime(s)
        return dt.strftime("%Y-%m")
    except:
        return s

def clean_units_sold(val):
    """
    Cleans Units_Sold field: handles commas ('9,600'), dashes ('-'), strings.
    Returns float or np.nan.
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s in ['-', '', 'none', 'null', 'nan']:
        return 0.0
    # Remove commas and spaces
    s = s.replace(',', '').replace(' ', '')
    try:
        return float(s)
    except:
        return np.nan

class DataCleaner:
    def __init__(self, train_path=TRAIN_FILE, test_path=TEST_FILE, 
                 macro_path=MACRO_FILE, sku_master_path=SKU_MASTER_FILE):
        self.train_path = train_path
        self.test_path = test_path
        self.macro_path = macro_path
        self.sku_master_path = sku_master_path
        self.audit_stats = {}

    def run_cleaning(self):
        print(">>> Starting Data Cleaning Pipeline...")
        # 1. Load Raw Data
        df_train_raw = pd.read_csv(self.train_path)
        df_test_raw = pd.read_csv(self.test_path)
        df_sku_master = pd.read_csv(self.sku_master_path)
        df_macro = pd.read_csv(self.macro_path)

        initial_train_rows = len(df_train_raw)
        initial_nulls = df_train_raw.isnull().sum().to_dict()
        
        # 2. De-duplicate Record_ID using latest Load_Timestamp (Rule 2)
        print("-> Resolving duplicate Record_IDs via latest Load_Timestamp...")
        df_train = df_train_raw.copy()
        df_train['Load_Timestamp'] = pd.to_datetime(df_train['Load_Timestamp'], errors='coerce')
        # Sort so latest timestamp comes first
        df_train = df_train.sort_values(by=['Record_ID', 'Load_Timestamp'], ascending=[True, False])
        dup_count = df_train['Record_ID'].duplicated().sum()
        df_train = df_train.drop_duplicates(subset=['Record_ID'], keep='first').reset_index(drop=True)
        print(f"   Removed {dup_count} duplicate records. Retained {len(df_train)} rows.")

        # 3. Robust Date Normalization & Rule 1: Trust Date over Year/Month
        print("-> Normalizing Dates and resolving Year/Month mismatches...")
        df_train['Date'] = df_train['Date'].apply(parse_date_robust)
        df_train['Year'] = df_train['Date'].str.slice(0, 4).astype(int)
        df_train['Month'] = df_train['Date'].str.slice(5, 7).astype(int)

        # Same for test
        df_test = df_test_raw.copy()
        df_test['Date'] = df_test['Date'].apply(parse_date_robust)
        df_test['Year'] = df_test['Date'].str.slice(0, 4).astype(int)
        df_test['Month'] = df_test['Date'].str.slice(5, 7).astype(int)

        # 4. Clean Units_Sold
        print("-> Standardizing Units_Sold and resolving non-numeric strings...")
        raw_units_str_fixes = df_train['Units_Sold'].astype(str).str.contains(',|-', regex=True).sum()
        df_train['Units_Sold_Clean'] = df_train['Units_Sold'].apply(clean_units_sold)

        # 5. Missing Metadata Imputation via SKU Master
        print("-> Imputing missing metadata from SKU Master...")
        sku_fam_map = df_sku_master.set_index('SKU_ID')['Product_Family'].to_dict()
        sku_bu_map = df_sku_master.set_index('SKU_ID')['Business_Unit'].to_dict()
        sku_desc_map = df_sku_master.set_index('SKU_ID')['SKU_Description'].to_dict()

        fam_nulls_before = df_train['Product_Family'].isna().sum()
        df_train['Product_Family'] = df_train['Product_Family'].fillna(df_train['SKU_ID'].map(sku_fam_map))
        df_train['Business_Unit'] = df_train['Business_Unit'].fillna(df_train['SKU_ID'].map(sku_bu_map))
        df_train['SKU_Description'] = df_train['SKU_Description'].fillna(df_train['SKU_ID'].map(sku_desc_map))

        # Impute missing Region if customer has exclusive region mapping, else forward fill by SKU-Customer
        reg_nulls_before = df_train['Region'].isna().sum()
        cust_region_mode = df_train.dropna(subset=['Region']).groupby('Customer_Name')['Region'].agg(lambda x: x.mode()[0]).to_dict()
        df_train['Region'] = df_train['Region'].fillna(df_train['Customer_Name'].map(cust_region_mode))
        # If any region still null, use global mode
        df_train['Region'] = df_train['Region'].fillna('North America')

        # 6. Price Reconciliation & Missing Units Recovery
        print("-> Reconciling Unit Prices and recovering missing Units_Sold...")
        # Step A: Recover missing Units_Sold where Revenue and Price exist
        missing_units_mask = df_train['Units_Sold_Clean'].isna()
        recoverable_mask = missing_units_mask & df_train['Revenue_USD'].notna() & (df_train['Unit_Price_USD'] > 0)
        recovered_count = recoverable_mask.sum()
        df_train.loc[recoverable_mask, 'Units_Sold_Clean'] = np.round(
            df_train.loc[recoverable_mask, 'Revenue_USD'] / df_train.loc[recoverable_mask, 'Unit_Price_USD']
        )
        # Any remaining null units are filled with 0 (no order recorded)
        df_train['Units_Sold_Clean'] = df_train['Units_Sold_Clean'].fillna(0.0)

        # Step B: Identify and fix decimal/unit anomalies in Unit_Price_USD
        # Compute annual median price per SKU
        sku_annual_price = df_train[df_train['Unit_Price_USD'] > 0].groupby(['SKU_ID', 'Year'])['Unit_Price_USD'].median().reset_index()
        sku_annual_price = sku_annual_price.rename(columns={'Unit_Price_USD': 'Annual_Median_Price'})
        df_train = df_train.merge(sku_annual_price, on=['SKU_ID', 'Year'], how='left')

        # Fallback to overall SKU median if annual median missing
        sku_overall_price = df_train[df_train['Unit_Price_USD'] > 0].groupby('SKU_ID')['Unit_Price_USD'].median().to_dict()
        df_train['Annual_Median_Price'] = df_train['Annual_Median_Price'].fillna(df_train['SKU_ID'].map(sku_overall_price))

        # Fix 100x pricing scale errors (e.g. 67308 instead of 673.08)
        price_ratio = df_train['Unit_Price_USD'] / df_train['Annual_Median_Price'].replace(0, np.nan)
        scale_100x_mask = (price_ratio > 70) & (price_ratio < 130)
        df_train.loc[scale_100x_mask, 'Unit_Price_USD'] = df_train.loc[scale_100x_mask, 'Unit_Price_USD'] / 100.0

        scale_div100_mask = (price_ratio > 0.007) & (price_ratio < 0.013)
        df_train.loc[scale_div100_mask, 'Unit_Price_USD'] = df_train.loc[scale_div100_mask, 'Unit_Price_USD'] * 100.0

        # Fix missing Unit_Price_USD
        df_train['Unit_Price_USD'] = df_train['Unit_Price_USD'].fillna(df_train['Annual_Median_Price'])

        # Recompute Revenue_USD consistency
        df_train['Revenue_USD'] = df_train['Units_Sold_Clean'] * df_train['Unit_Price_USD']

        # Step C: Negative units treatment (Returns / Chargebacks)
        negative_units_count = (df_train['Units_Sold_Clean'] < 0).sum()
        df_train['Is_Return_Adjustment'] = (df_train['Units_Sold_Clean'] < 0).astype(int)
        df_train['Raw_Units_Sold'] = df_train['Units_Sold_Clean']
        # For demand forecasting, clipped to non-negative (Rule: customer unconstrained demand >= 0)
        df_train['Units_Sold'] = np.maximum(0.0, df_train['Units_Sold_Clean']).astype(int)

        # Drop temporary columns
        df_train = df_train.drop(columns=['Units_Sold_Clean', 'Annual_Median_Price'])

        # 7. Clean Macro Indicators
        print("-> Cleaning Macro Indicators (harmonizing scales & interpolating NaNs)...")
        df_macro_clean = df_macro.copy()
        df_macro_clean['Date'] = df_macro_clean['Date'].apply(parse_date_robust)
        df_macro_clean['Year'] = df_macro_clean['Date'].str.slice(0, 4).astype(int)
        df_macro_clean['Month'] = df_macro_clean['Date'].str.slice(5, 7).astype(int)

        # Scale 2019 Auto_Production_YoY_Pct from decimal to percentage
        mask_2019 = df_macro_clean['Year'] == 2019
        df_macro_clean.loc[mask_2019, 'Auto_Production_YoY_Pct'] = df_macro_clean.loc[mask_2019, 'Auto_Production_YoY_Pct'] * 100.0

        # Interpolate numeric columns linearly
        num_macro_cols = [c for c in df_macro_clean.columns if c not in ['Date', 'Year', 'Month']]
        df_macro_clean[num_macro_cols] = df_macro_clean[num_macro_cols].interpolate(method='linear', limit_direction='both')

        # 8. Save Cleaned Datasets
        print(f"-> Saving cleaned train to {CLEANED_TRAIN_FILE}...")
        df_train.to_csv(CLEANED_TRAIN_FILE, index=False)
        print(f"-> Saving cleaned test to {CLEANED_TEST_FILE}...")
        df_test.to_csv(CLEANED_TEST_FILE, index=False)
        macro_clean_path = os.path.join(os.path.dirname(CLEANED_TRAIN_FILE), "cleaned_macro_indicators.csv")
        df_macro_clean.to_csv(macro_clean_path, index=False)

        # 9. Generate Multi-Tab Data Quality Report (Excel)
        print(f"-> Generating Data Quality Report at {DATA_QUALITY_REPORT_FILE}...")
        summary_df = pd.DataFrame([
            {"Metric": "Initial Raw Rows", "Value": initial_train_rows},
            {"Metric": "Duplicates Removed", "Value": dup_count},
            {"Metric": "Cleaned Rows Retained", "Value": len(df_train)},
            {"Metric": "Non-numeric/Comma Units Fixed", "Value": raw_units_str_fixes},
            {"Metric": "Missing Units Recovered via Price/Rev", "Value": recovered_count},
            {"Metric": "Product_Family Nulls Imputed", "Value": fam_nulls_before},
            {"Metric": "Region Nulls Imputed", "Value": reg_nulls_before},
            {"Metric": "100x Price Scale Anomalies Fixed", "Value": scale_100x_mask.sum()},
            {"Metric": "Negative Unit Adjustments Flagged", "Value": negative_units_count},
            {"Metric": "2019 Macro Scale Normalization", "Value": mask_2019.sum()}
        ])

        with pd.ExcelWriter(DATA_QUALITY_REPORT_FILE, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name="Cleaning_Summary", index=False)
            pd.DataFrame([
                {"Column": col, "Raw_Nulls": initial_nulls.get(col, 0), "Clean_Nulls": df_train[col].isna().sum() if col in df_train else 0}
                for col in initial_nulls
            ]).to_excel(writer, sheet_name="Null_Comparison", index=False)
            df_train[df_train['Is_Return_Adjustment'] == 1].head(100).to_excel(writer, sheet_name="Return_Adjustments", index=False)

        print(">>> Data Cleaning Pipeline Completed Successfully!\n")
        return df_train, df_test, df_macro_clean

if __name__ == '__main__':
    cleaner = DataCleaner()
    cleaner.run_cleaning()
