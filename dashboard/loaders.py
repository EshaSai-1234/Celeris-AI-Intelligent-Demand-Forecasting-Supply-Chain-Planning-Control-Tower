from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import (
    ANOMALY_REGISTER,
    CLEANED_TEST,
    CLEANED_TRAIN,
    FORECAST_DETAIL,
    INVENTORY_REC,
    MODEL_COMPARISON,
    OUTPUT_DIR,
    QUALITY_REPORT,
    SHAP_GLOBAL,
    SHAP_SKU,
    SUBMISSION_CSV,
)


def _read(path: Path, **kwargs):
    if not path.exists():
        return None
    if path.suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    if path.suffix == ".xlsx":
        return pd.read_excel(path, **kwargs)
    if path.suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    return None


@st.cache_data(show_spinner=False)
def load_all() -> dict:
    return {
        "train": _read(CLEANED_TRAIN),
        "test": _read(CLEANED_TEST),
        "forecast": _read(FORECAST_DETAIL),
        "submission": _read(SUBMISSION_CSV),
        "anomalies": _read(ANOMALY_REGISTER),
        "inventory": _read(INVENTORY_REC),
        "metrics": _read(MODEL_COMPARISON),
        "shap_global": _read(SHAP_GLOBAL),
        "shap_sku": _read(SHAP_SKU),
        "macro": _read(OUTPUT_DIR / "cleaned_macro.csv"),
        "events": _read(OUTPUT_DIR / "cleaned_events.csv"),
        "stockouts": _read(OUTPUT_DIR / "cleaned_stockouts.csv"),
        "sku": _read(OUTPUT_DIR / "cleaned_sku_master.csv"),
        "quality": _read(QUALITY_REPORT, sheet_name="summary") if QUALITY_REPORT.exists() else None,
        "notes": _read(OUTPUT_DIR / "demand_reconstruction_notes.csv"),
    }


def apply_filters(df: pd.DataFrame, family, sku, region, channel, customer) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df
    if family and "Product_Family" in out.columns:
        out = out[out["Product_Family"].isin(family)]
    if sku and "SKU_ID" in out.columns:
        out = out[out["SKU_ID"].isin(sku)]
    if region and "Region" in out.columns:
        out = out[out["Region"].isin(region)]
    if channel and "Sales_Channel" in out.columns:
        out = out[out["Sales_Channel"].isin(channel)]
    if customer and "Customer_Name" in out.columns:
        out = out[out["Customer_Name"].isin(customer)]
    return out


def need_outputs() -> bool:
    return not FORECAST_DETAIL.exists()
