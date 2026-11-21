"""Schema validation, cleaning and data-quality reporting."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from src.config import (
    CLEANED_TEST,
    CLEANED_TRAIN,
    EVENTS_CSV,
    MACRO_CSV,
    OUTPUT_DIR,
    QUALITY_REPORT,
    SKU_MASTER_CSV,
    STOCKOUTS_CSV,
    TEST_CSV,
    TRAIN_CSV,
)
from src.utils import (
    normalize_channel,
    normalize_customer,
    normalize_region,
    normalize_sku_id,
    parse_units,
    resolve_period_from_row,
)


def load_raw() -> dict[str, pd.DataFrame]:
    return {
        "train": pd.read_csv(TRAIN_CSV),
        "test": pd.read_csv(TEST_CSV),
        "sku": pd.read_csv(SKU_MASTER_CSV),
        "macro": pd.read_csv(MACRO_CSV),
        "events": pd.read_csv(EVENTS_CSV),
        "stockouts": pd.read_csv(STOCKOUTS_CSV),
    }


def _apply_period(df: pd.DataFrame) -> pd.DataFrame:
    dates = df["Date"] if "Date" in df.columns else pd.Series([np.nan] * len(df))
    years = df["Year"] if "Year" in df.columns else pd.Series([np.nan] * len(df))
    months = df["Month"] if "Month" in df.columns else pd.Series([np.nan] * len(df))
    triples = [
        resolve_period_from_row(d, y, m) for d, y, m in zip(dates.tolist(), years.tolist(), months.tolist())
    ]
    resolved = pd.DataFrame(triples, columns=["Period", "Period_Source"])
    out = pd.concat([df.reset_index(drop=True), resolved], axis=1)
    parts = out["Period"].astype(str).str.split("-", expand=True)
    out["Year_Resolved"] = pd.to_numeric(parts[0], errors="coerce")
    out["Month_Resolved"] = pd.to_numeric(parts[1], errors="coerce")
    return out


def clean_sku_master(sku: pd.DataFrame) -> pd.DataFrame:
    out = sku.copy()
    out["SKU_ID"] = out["SKU_ID"].map(lambda x: normalize_sku_id(x, None))
    out["Launch_Period"] = out["Launch_Date"].map(
        lambda x: resolve_period_from_row(x, None, None)[0]
    )
    out["EOL_Period"] = out["EOL_Date"].map(
        lambda x: resolve_period_from_row(x, None, None)[0] if pd.notna(x) else np.nan
    )
    out["Replaces_SKU"] = out["Replaces_SKU"].map(
        lambda x: normalize_sku_id(x, None) if pd.notna(x) else np.nan
    )
    return out


def clean_train(train: pd.DataFrame, sku: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    stats: dict = {}
    df = train.copy()
    stats["raw_rows"] = int(len(df))
    valid = set(sku["SKU_ID"])

    df["SKU_ID_Raw"] = df["SKU_ID"]
    df["SKU_ID"] = df["SKU_ID"].map(lambda x: normalize_sku_id(x, valid))
    df["SKU_In_Master"] = df["SKU_ID"].isin(valid)

    df["Region_Raw"] = df["Region"]
    df["Region"] = df["Region"].map(normalize_region)
    df["Sales_Channel"] = df["Sales_Channel"].map(normalize_channel)
    df["Customer_Name"] = df["Customer_Name"].map(normalize_customer)

    sku_map = sku.set_index("SKU_ID")
    miss_fam = df["Product_Family"].isna() & df["SKU_In_Master"]
    df.loc[miss_fam, "Product_Family"] = df.loc[miss_fam, "SKU_ID"].map(sku_map["Product_Family"])
    miss_desc = df["SKU_Description"].isna() & df["SKU_In_Master"]
    df.loc[miss_desc, "SKU_Description"] = df.loc[miss_desc, "SKU_ID"].map(sku_map["SKU_Description"])
    miss_bu = df["Business_Unit"].isna() & df["SKU_In_Master"] if "Business_Unit" in df else None
    if miss_bu is not None:
        df.loc[miss_bu, "Business_Unit"] = df.loc[miss_bu, "SKU_ID"].map(sku_map["Business_Unit"])

    df = _apply_period(df)
    df["Units_Sold_Raw"] = df["Units_Sold"]
    df["Units_Sold"] = df["Units_Sold"].map(parse_units)
    df["Unit_Price_USD"] = pd.to_numeric(df["Unit_Price_USD"], errors="coerce")
    df["Revenue_USD"] = pd.to_numeric(df["Revenue_USD"], errors="coerce")

    recoverable = df["Units_Sold"].isna() & df["Revenue_USD"].notna() & (df["Unit_Price_USD"] > 0)
    df.loc[recoverable, "Units_Sold"] = (
        df.loc[recoverable, "Revenue_USD"] / df.loc[recoverable, "Unit_Price_USD"]
    )
    stats["units_imputed_from_revenue"] = int(recoverable.sum())

    rev_from_units = df["Revenue_USD"].isna() & df["Units_Sold"].notna() & df["Unit_Price_USD"].notna()
    df.loc[rev_from_units, "Revenue_USD"] = (
        df.loc[rev_from_units, "Units_Sold"] * df.loc[rev_from_units, "Unit_Price_USD"]
    )
    stats["revenue_imputed_from_units"] = int(rev_from_units.sum())

    df["Load_Timestamp"] = pd.to_datetime(df["Load_Timestamp"], errors="coerce")
    before = len(df)
    df = df.sort_values(["Record_ID", "Load_Timestamp"])
    df = df.drop_duplicates("Record_ID", keep="last")
    stats["duplicate_record_ids_dropped"] = before - len(df)

    grain_keys = ["Period", "SKU_ID", "Region", "Customer_Name", "Sales_Channel"]
    extra = df.duplicated(grain_keys, keep=False)
    stats["remaining_grain_duplicate_rows"] = int(extra.sum())
    if extra.any():
        df = df.sort_values(grain_keys + ["Load_Timestamp"]).drop_duplicates(grain_keys, keep="last")

    df["Date"] = df["Period"]
    df["Year"] = df["Year_Resolved"]
    df["Month"] = df["Month_Resolved"]
    stats["cleaned_rows"] = int(len(df))
    stats["sku_not_in_master"] = int((~df["SKU_In_Master"]).sum())
    stats["unparseable_period"] = int(df["Period"].isna().sum())
    stats["region_still_unknown"] = int(df["Region"].isna().sum())
    stats["units_still_null"] = int(df["Units_Sold"].isna().sum())
    stats["negative_units"] = int((df["Units_Sold"] < 0).sum())
    return df, stats


def clean_test(test: pd.DataFrame, sku: pd.DataFrame) -> pd.DataFrame:
    df = test.copy()
    df["submission_row"] = np.arange(len(df), dtype=int)
    valid = set(sku["SKU_ID"])
    df["SKU_ID"] = df["SKU_ID"].map(lambda x: normalize_sku_id(x, valid))
    df["Region"] = df["Region"].map(normalize_region)
    df["Sales_Channel"] = df["Sales_Channel"].map(normalize_channel)
    df["Customer_Name"] = df["Customer_Name"].map(normalize_customer)
    df = _apply_period(df)
    df["Date"] = df["Period"]
    df["Year"] = df["Year_Resolved"]
    df["Month"] = df["Month_Resolved"]
    return df


def clean_macro(macro: pd.DataFrame) -> pd.DataFrame:
    df = _apply_period(macro.copy())
    df["Date"] = df["Period"]
    num_cols = [
        "ISM_Manufacturing_PMI",
        "Auto_Production_YoY_Pct",
        "Vehicle_Inventory_Days",
        "Raw_Material_Cost_Index",
        "New_Vehicle_Orders_Index",
        "Industrial_Util_Rate_Pct",
        "Fuel_Price_Index",
        "EV_Adoption_Index",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].interpolate(limit_direction="both")
    # Observed scale break: 2019 Auto_Production_YoY_Pct is ~ -0.02 while 2020+ is percent points.
    early = df["Year_Resolved"] <= 2019
    if early.any() and df.loc[~early, "Auto_Production_YoY_Pct"].abs().median() > 1:
        df.loc[early, "Auto_Production_YoY_Pct"] = df.loc[early, "Auto_Production_YoY_Pct"] * 100.0
        df["Auto_Production_YoY_Rescaled_2019"] = early.astype(int)
    return df


def clean_events(events: pd.DataFrame) -> pd.DataFrame:
    df = events.copy()
    df["Start_Period"] = df["Start_Date"].map(lambda x: resolve_period_from_row(x, None, None)[0])
    df["End_Period"] = df.apply(
        lambda r: resolve_period_from_row(r["End_Date"], None, None)[0]
        if pd.notna(r["End_Date"])
        else r["Start_Period"],
        axis=1,
    )
    return df


def clean_stockouts(stock: pd.DataFrame, sku: pd.DataFrame) -> pd.DataFrame:
    df = stock.copy()
    valid = set(sku["SKU_ID"])
    df["SKU_ID"] = df["SKU_ID"].map(lambda x: normalize_sku_id(x, valid))
    df["Region"] = df["Region"].map(normalize_region)
    df = _apply_period(df)
    df["Date"] = df["Period"]
    for c in [
        "Requested_Units",
        "Carried_In_Units",
        "Shipped_Units",
        "Backordered_Units",
        "Lost_Units",
        "Fill_Rate_Pct",
        "Stockout_Days",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_quality_report(
    raw: dict[str, pd.DataFrame],
    train_c: pd.DataFrame,
    test_c: pd.DataFrame,
    train_stats: dict,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    issues = []

    def add(sheet, issue, count, detail):
        issues.append({"sheet": sheet, "issue": issue, "count": count, "detail": detail})

    add("train", "raw_rows", train_stats["raw_rows"], "as loaded")
    add("train", "cleaned_rows", train_stats["cleaned_rows"], "after Record_ID and grain dedupe")
    add("train", "duplicate_record_ids_dropped", train_stats["duplicate_record_ids_dropped"], "kept latest Load_Timestamp")
    add("train", "units_imputed_from_revenue", train_stats["units_imputed_from_revenue"], "Revenue/Unit_Price where Units_Sold missing")
    add("train", "sku_not_in_master_after_norm", train_stats["sku_not_in_master"], "normalized SKU still absent from sku_master")
    add("train", "unparseable_period", train_stats["unparseable_period"], "")
    add("train", "region_null", train_stats["region_still_unknown"], "left null; not filled with a fabricated region")
    add("train", "units_null", train_stats["units_still_null"], "not filled with zero")
    add("train", "negative_units", train_stats["negative_units"], "retained and flagged")

    src = train_c["Period_Source"].value_counts()
    for k, v in src.items():
        add("train", f"period_source_{k}", int(v), "")

    mismatch = (train_c["Period_Source"] == "date_over_year_month").sum()
    add("train", "date_trusted_over_year_month", int(mismatch), "PROJECT_SPEC: trust Date")

    calc = train_c["Units_Sold"] * train_c["Unit_Price_USD"]
    valid = train_c["Units_Sold"].notna() & train_c["Unit_Price_USD"].notna() & train_c["Revenue_USD"].notna()
    rev_err = (calc[valid] - train_c.loc[valid, "Revenue_USD"]).abs() > 0.05
    add("train", "revenue_inconsistent_gt_0.05", int(rev_err.sum()), "|units*price - revenue| > 0.05")

    add("test", "rows", int(len(test_c)), "row order preserved via submission_row")
    add("test", "unique_date_sku_region", int(test_c[["Date", "SKU_ID", "Region"]].drop_duplicates().shape[0]), "")

    overview = pd.DataFrame(issues)
    missing_train = raw["train"].isna().sum().rename("null_count").reset_index().rename(columns={"index": "column"})
    missing_train["null_pct"] = missing_train["null_count"] / len(raw["train"])
    region_map = (
        train_c.groupby(["Region_Raw", "Region"], dropna=False).size().reset_index(name="rows")
        if "Region_Raw" in train_c
        else pd.DataFrame()
    )
    unit_flags = pd.DataFrame(
        {
            "metric": ["zero_units", "negative_units", "null_units", "price_zero"],
            "count": [
                int((train_c["Units_Sold"] == 0).sum()),
                int((train_c["Units_Sold"] < 0).sum()),
                int(train_c["Units_Sold"].isna().sum()),
                int((train_c["Unit_Price_USD"] == 0).sum()),
            ],
        }
    )
    period_hist = train_c["Period"].value_counts().sort_index().rename("rows").reset_index()

    with pd.ExcelWriter(QUALITY_REPORT, engine="openpyxl") as writer:
        overview.to_excel(writer, sheet_name="summary", index=False)
        missing_train.to_excel(writer, sheet_name="train_raw_nulls", index=False)
        unit_flags.to_excel(writer, sheet_name="units_flags", index=False)
        period_hist.to_excel(writer, sheet_name="rows_by_period", index=False)
        if len(region_map):
            region_map.to_excel(writer, sheet_name="region_normalization", index=False)
        pd.DataFrame(list(train_stats.items()), columns=["stat", "value"]).to_excel(
            writer, sheet_name="clean_stats", index=False
        )


def run_cleaning() -> dict[str, pd.DataFrame]:
    raw = load_raw()
    sku = clean_sku_master(raw["sku"])
    train_c, stats = clean_train(raw["train"], sku)
    test_c = clean_test(raw["test"], sku)
    macro = clean_macro(raw["macro"])
    events = clean_events(raw["events"])
    stock = clean_stockouts(raw["stockouts"], sku)
    build_quality_report(raw, train_c, test_c, stats)

    keep_train = [
        "Record_ID",
        "Date",
        "Year",
        "Month",
        "Period",
        "Period_Source",
        "Product_Family",
        "Business_Unit",
        "SKU_ID",
        "SKU_Description",
        "Customer_Name",
        "Sales_Channel",
        "Region",
        "Units_Sold",
        "Unit_Price_USD",
        "Revenue_USD",
        "Load_Timestamp",
        "SKU_In_Master",
        "Units_Sold_Raw",
    ]
    train_c[keep_train].to_csv(CLEANED_TRAIN, index=False)
    test_c.to_csv(CLEANED_TEST, index=False)
    sku.to_csv(OUTPUT_DIR / "cleaned_sku_master.csv", index=False)
    macro.to_csv(OUTPUT_DIR / "cleaned_macro.csv", index=False)
    events.to_csv(OUTPUT_DIR / "cleaned_events.csv", index=False)
    stock.to_csv(OUTPUT_DIR / "cleaned_stockouts.csv", index=False)
    return {
        "train": train_c,
        "test": test_c,
        "sku": sku,
        "macro": macro,
        "events": events,
        "stockouts": stock,
        "stats": stats,
    }
