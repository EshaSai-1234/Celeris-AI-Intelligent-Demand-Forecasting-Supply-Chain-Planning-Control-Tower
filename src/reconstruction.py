"""Unconstrained demand reconstruction from observed sales and documented stockouts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import OUTPUT_DIR, PANEL_PATH, RECONSTRUCTION_NOTES, TRAIN_END, TRAIN_START
from src.utils import period_range


def aggregate_train_panel(train: pd.DataFrame, sku: pd.DataFrame) -> pd.DataFrame:
    df = train.dropna(subset=["Period", "SKU_ID"]).copy()
    df = df[df["Units_Sold"].isna() | (df["Units_Sold"] >= 0)]
    g = (
        df.groupby(["Period", "SKU_ID", "Region"], dropna=False)
        .agg(
            Units_Sold_Observed=("Units_Sold", "sum"),
            Revenue_USD=("Revenue_USD", "sum"),
            Unit_Price_USD=("Unit_Price_USD", "median"),
            Product_Family=("Product_Family", "first"),
            Business_Unit=("Business_Unit", "first"),
            Customer_Name=("Customer_Name", lambda s: s.dropna().mode().iloc[0] if len(s.dropna()) else np.nan),
            Sales_Channel=("Sales_Channel", lambda s: s.dropna().mode().iloc[0] if len(s.dropna()) else np.nan),
            n_source_rows=("Record_ID", "count"),
        )
        .reset_index()
    )
    g = g.merge(
        sku[
            [
                "SKU_ID",
                "Size_Class",
                "Powertrain_Application",
                "Launch_Period",
                "EOL_Period",
                "Replaces_SKU",
                "Lifecycle_Status",
                "Product_Family",
                "Business_Unit",
            ]
        ],
        on="SKU_ID",
        how="left",
        suffixes=("", "_master"),
    )
    g["Product_Family"] = g["Product_Family"].fillna(g["Product_Family_master"])
    g["Business_Unit"] = g["Business_Unit"].fillna(g["Business_Unit_master"])
    g = g.drop(columns=[c for c in g.columns if c.endswith("_master")])
    return g


def reconstruct_demand(panel: pd.DataFrame, stockouts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    so = stockouts.copy()
    so_keys = ["Period", "SKU_ID", "Region"]
    keep = so_keys + [
        "Requested_Units",
        "Shipped_Units",
        "Backordered_Units",
        "Lost_Units",
        "Fill_Rate_Pct",
        "Stockout_Days",
        "Reason",
    ]
    so = so[keep].drop_duplicates(so_keys)
    merged = panel.merge(so, on=so_keys, how="left", indicator=True)
    merged["has_stockout_record"] = merged["_merge"].eq("both")
    merged = merged.drop(columns="_merge")

    lost = merged["Lost_Units"].fillna(0).clip(lower=0)
    observed = merged["Units_Sold_Observed"]
    unconstrained = observed.copy()
    documented = merged["has_stockout_record"] & observed.notna()
    unconstrained.loc[documented] = observed.loc[documented] + lost.loc[documented]

    # If stockout exists, observed is null, but shipped units are present, use shipped + lost.
    fill_obs = merged["has_stockout_record"] & observed.isna() & merged["Shipped_Units"].notna()
    unconstrained.loc[fill_obs] = merged.loc[fill_obs, "Shipped_Units"] + lost.loc[fill_obs]

    merged["Demand_Unconstrained"] = unconstrained
    merged["Lost_Sales_Used"] = np.where(documented | fill_obs, lost, 0.0)
    merged["Reconstruction_Confidence"] = np.select(
        [
            documented & (lost > 0),
            merged["has_stockout_record"] & (lost == 0),
            observed.notna(),
        ],
        ["high_documented_lost_units", "stockout_without_lost_units", "observed_only"],
        default="missing_not_zero_filled",
    )

    notes = pd.DataFrame(
        [
            {
                "assumption_id": "R1",
                "statement": "Units_Sold=0 is treated as a genuine no-order, not as missing demand.",
                "source": "PROJECT_SPEC.md",
            },
            {
                "assumption_id": "R2",
                "statement": "Missing SKU-region-month rows are not filled with zero.",
                "source": "PROJECT_SPEC.md",
            },
            {
                "assumption_id": "R3",
                "statement": "Unconstrained demand = observed Units_Sold + Lost_Units only when inventory_stockouts has a matching Date-SKU-Region row. No extra lost sales are invented.",
                "source": "inventory_stockouts.csv",
            },
            {
                "assumption_id": "R4",
                "statement": "Negative parsed Units_Sold rows are excluded from the panel rather than clipped.",
                "source": "cleaning",
            },
            {
                "assumption_id": "R5",
                "statement": "When a stockout row exists and observed sales are missing, Shipped_Units + Lost_Units is used if Shipped_Units is present.",
                "source": "inventory_stockouts.csv",
            },
        ]
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    notes.to_csv(RECONSTRUCTION_NOTES, index=False)
    merged.to_csv(PANEL_PATH, index=False)
    return merged, notes


def attach_test_keys(panel: pd.DataFrame, test: pd.DataFrame, sku: pd.DataFrame) -> pd.DataFrame:
    """Ensure every test SKU-region exists in the modeling universe without fabricating history."""
    keys = test[["SKU_ID", "Region", "Product_Family", "Business_Unit", "Customer_Name", "Sales_Channel"]].drop_duplicates()
    hist_keys = panel[["SKU_ID", "Region"]].drop_duplicates()
    missing = keys.merge(hist_keys, on=["SKU_ID", "Region"], how="left", indicator=True)
    missing = missing[missing["_merge"] == "left_only"].drop(columns="_merge")
    if missing.empty:
        return panel
    extra = missing.merge(sku, on="SKU_ID", how="left", suffixes=("", "_m"))
    extra["Period"] = np.nan
    extra["Units_Sold_Observed"] = np.nan
    extra["Demand_Unconstrained"] = np.nan
    extra["has_stockout_record"] = False
    extra["Reconstruction_Confidence"] = "no_history"
    # Keep columns aligned
    for c in panel.columns:
        if c not in extra:
            extra[c] = np.nan
    extra = extra[panel.columns]
    return pd.concat([panel, extra], ignore_index=True)
