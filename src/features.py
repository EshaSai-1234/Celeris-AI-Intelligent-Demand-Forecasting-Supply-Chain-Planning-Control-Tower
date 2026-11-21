"""Leakage-safe feature engineering for SKU-region monthly series."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import LAGS, OUTPUT_DIR, ROLL_WINDOWS
from src.utils import months_between, period_range, period_to_timestamp


MACRO_COLS = [
    "ISM_Manufacturing_PMI",
    "Auto_Production_YoY_Pct",
    "Vehicle_Inventory_Days",
    "Raw_Material_Cost_Index",
    "New_Vehicle_Orders_Index",
    "Industrial_Util_Rate_Pct",
    "Fuel_Price_Index",
    "EV_Adoption_Index",
]


def _expand_events(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in events.itertuples(index=False):
        start = r.Start_Period
        end = r.End_Period if pd.notna(r.End_Period) else start
        if pd.isna(start):
            continue
        end = start if pd.isna(end) else end
        if str(end) < str(start):
            end = start
        for p in period_range(str(start), str(end)):
            rows.append(
                {
                    "Period": p,
                    "Scope_Type": r.Scope_Type,
                    "Scope_Value": r.Scope_Value,
                    "Event_Type": r.Event_Type,
                }
            )
    return pd.DataFrame(rows)


def event_flags(panel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    ev = _expand_events(events)
    if ev.empty:
        panel["event_holiday"] = 0
        panel["event_disruption"] = 0
        panel["event_price_change"] = 0
        panel["event_supply_constraint"] = 0
        panel["event_any"] = 0
        return panel
    ev["event_holiday"] = ev["Event_Type"].eq("Holiday").astype(int)
    ev["event_disruption"] = ev["Event_Type"].eq("Disruption").astype(int)
    ev["event_price_change"] = ev["Event_Type"].eq("Price Change").astype(int)
    ev["event_supply_constraint"] = ev["Event_Type"].isin(["Supply Constraint", "Logistics"]).astype(int)

    global_ev = (
        ev[ev["Scope_Type"].eq("Global")]
        .groupby("Period")[["event_holiday", "event_disruption", "event_price_change", "event_supply_constraint"]]
        .max()
        .add_prefix("g_")
        .reset_index()
    )
    region_ev = ev[ev["Scope_Type"].eq("Region")].copy()
    region_ev["Region"] = region_ev["Scope_Value"]
    region_ev = (
        region_ev.groupby(["Period", "Region"])[
            ["event_holiday", "event_disruption", "event_price_change", "event_supply_constraint"]
        ]
        .max()
        .add_prefix("r_")
        .reset_index()
    )
    fam_ev = ev[ev["Scope_Type"].eq("Product_Family")].copy()
    fam_ev["Product_Family"] = fam_ev["Scope_Value"]
    fam_ev = (
        fam_ev.groupby(["Period", "Product_Family"])[
            ["event_holiday", "event_disruption", "event_price_change", "event_supply_constraint"]
        ]
        .max()
        .add_prefix("f_")
        .reset_index()
    )

    out = panel.merge(global_ev, on="Period", how="left")
    out = out.merge(region_ev, on=["Period", "Region"], how="left")
    if "Product_Family" in out.columns and len(fam_ev):
        out = out.merge(fam_ev, on=["Period", "Product_Family"], how="left")
    for prefix in ("g_", "r_", "f_"):
        for col in ["event_holiday", "event_disruption", "event_price_change", "event_supply_constraint"]:
            c = prefix + col
            if c not in out:
                out[c] = 0
            out[c] = out[c].fillna(0)
    out["event_holiday"] = out[["g_event_holiday", "r_event_holiday"]].max(axis=1)
    out["event_disruption"] = out[["g_event_disruption", "r_event_disruption"]].max(axis=1)
    out["event_price_change"] = out["g_event_price_change"]
    out["event_supply_constraint"] = out[[c for c in out.columns if c.endswith("event_supply_constraint")]].max(axis=1)
    out["event_any"] = (
        out[["event_holiday", "event_disruption", "event_price_change", "event_supply_constraint"]].max(axis=1)
    )
    return out


def add_macro_lags(panel: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    m = macro[["Period"] + [c for c in MACRO_COLS if c in macro.columns]].drop_duplicates("Period").sort_values("Period")
    m = m.rename(columns={"Period": "macro_period"})
    out = panel.copy()
    # lag 1/2/3 only — contemporaneous macro for a future month is treated as not known at origin.
    for lag in (1, 2, 3):
        shifted = m.copy()
        shifted["Period"] = shifted["macro_period"].map(lambda p: _shift_period(p, lag))
        renamed = {c: f"{c}_lag{lag}" for c in MACRO_COLS if c in shifted}
        shifted = shifted.rename(columns=renamed).drop(columns=["macro_period"])
        out = out.merge(shifted, on="Period", how="left")
    return out


def _shift_period(period: str, lag: int) -> str:
    ts = period_to_timestamp(str(period)) + pd.DateOffset(months=lag)
    return f"{ts.year:04d}-{ts.month:02d}"


def add_calendar_lifecycle(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["Period"] + "-01", errors="coerce")
    out["month_num"] = ts.dt.month
    out["quarter"] = ts.dt.quarter
    out["year_num"] = ts.dt.year
    out["month_sin"] = np.sin(2 * np.pi * out["month_num"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month_num"] / 12)
    out["is_january"] = (out["month_num"] == 1).astype(int)

    launch_ts = pd.to_datetime(out["Launch_Period"].astype(str) + "-01", errors="coerce")
    eol_ts = pd.to_datetime(out["EOL_Period"].astype(str) + "-01", errors="coerce")
    out["sku_age_months"] = (ts.dt.year - launch_ts.dt.year) * 12 + (ts.dt.month - launch_ts.dt.month)
    out["months_to_eol"] = (eol_ts.dt.year - ts.dt.year) * 12 + (eol_ts.dt.month - ts.dt.month)
    out["is_new_product"] = (out["sku_age_months"].fillna(99) <= 6).astype(int)
    out["is_pre_launch"] = (out["sku_age_months"].fillna(0) < 0).astype(int)
    out["is_past_eol"] = (out["months_to_eol"].fillna(1) < 0).astype(int)
    out["eol_within_6m"] = ((out["months_to_eol"] >= 0) & (out["months_to_eol"] <= 6)).astype(int)
    return out


def add_lags_and_rolls(df: pd.DataFrame, target_col: str = "Demand_Unconstrained") -> pd.DataFrame:
    out = df.sort_values(["SKU_ID", "Region", "Period"]).copy()
    g = out.groupby(["SKU_ID", "Region"], sort=False)
    for lag in LAGS:
        out[f"{target_col}_lag{lag}"] = g[target_col].shift(lag)
    shifted = g[target_col].shift(1)
    for w in ROLL_WINDOWS:
        rolled = shifted.groupby([out["SKU_ID"], out["Region"]], sort=False)
        out[f"roll_mean_{w}"] = rolled.transform(lambda s, ww=w: s.rolling(ww, min_periods=max(1, ww // 2)).mean())
        out[f"roll_std_{w}"] = rolled.transform(lambda s, ww=w: s.rolling(ww, min_periods=max(1, ww // 2)).std())
    out["zero_share_12"] = shifted.groupby([out["SKU_ID"], out["Region"]], sort=False).transform(
        lambda s: (s.eq(0)).rolling(12, min_periods=4).mean()
    )
    fam_period = out.groupby(["Product_Family", "Region", "Period"], dropna=False)[target_col].transform("mean")
    out["_fam_tmp"] = fam_period
    out["family_region_mean_lag1"] = out.groupby(["SKU_ID", "Region"], sort=False)["_fam_tmp"].shift(1)
    out = out.drop(columns=["_fam_tmp"])
    return out


def successor_history(panel: pd.DataFrame, sku: pd.DataFrame) -> pd.DataFrame:
    """Copy predecessor history onto planned-launch successors that have no own sales."""
    repl = sku.dropna(subset=["Replaces_SKU"])[["SKU_ID", "Replaces_SKU", "Launch_Period"]]
    if repl.empty:
        return panel
    frames = [panel]
    existing = set(zip(panel["SKU_ID"], panel["Region"]))
    regions = panel["Region"].dropna().unique()
    for r in repl.itertuples(index=False):
        pred = panel[panel["SKU_ID"] == r.Replaces_SKU].copy()
        if pred.empty:
            continue
        for region in pred["Region"].dropna().unique():
            if (r.SKU_ID, region) in existing:
                continue
            chunk = pred[pred["Region"] == region].copy()
            chunk["SKU_ID"] = r.SKU_ID
            chunk["successor_from"] = r.Replaces_SKU
            frames.append(chunk)
            existing.add((r.SKU_ID, region))
    return pd.concat(frames, ignore_index=True)


def build_feature_table(
    panel: pd.DataFrame,
    macro: pd.DataFrame,
    events: pd.DataFrame,
    sku: pd.DataFrame,
) -> pd.DataFrame:
    df = panel[panel["Period"].notna()].copy()
    df = successor_history(df, sku)
    df = add_calendar_lifecycle(df)
    df = event_flags(df, events)
    df = add_macro_lags(df, macro)
    df = add_lags_and_rolls(df, "Demand_Unconstrained")
    df["target"] = df["Demand_Unconstrained"]
    df.to_parquet(OUTPUT_DIR / "feature_panel.parquet", index=False)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {
        "target",
        "Demand_Unconstrained",
        "Units_Sold_Observed",
        "Revenue_USD",
        "n_source_rows",
        "Requested_Units",
        "Shipped_Units",
        "Backordered_Units",
        "Lost_Units",
        "Fill_Rate_Pct",
        "Stockout_Days",
        "Reason",
        "Lost_Sales_Used",
        "Reconstruction_Confidence",
        "has_stockout_record",
        "Units_Sold_Raw",
        "Record_ID",
        "Load_Timestamp",
        "successor_from",
        "SKU_Description",
    }
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if df[c].dtype == object or str(df[c].dtype) == "category":
            cols.append(c)
        elif pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    # keep identifiers needed as cats
    return cols
