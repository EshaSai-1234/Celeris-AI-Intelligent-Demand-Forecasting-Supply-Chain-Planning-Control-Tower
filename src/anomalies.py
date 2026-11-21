"""Anomaly detection. Flags only; legitimate spikes are not deleted."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import ANOMALY_REGISTER, RANDOM_STATE


def _iqr_flags(series: pd.Series, k: float = 3.0) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        med = series.median()
        mad = (series - med).abs().median()
        if mad == 0 or pd.isna(mad):
            return pd.Series(False, index=series.index)
        z = 0.6745 * (series - med) / mad
        return z.abs() > 4
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return (series < lo) | (series > hi)


def detect_anomalies(panel: pd.DataFrame, stockouts: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    # Negative / non-numeric already parsed; flag remaining negatives and huge values.
    if "Units_Sold" in train.columns:
        neg = train.loc[train["Units_Sold"] < 0]
        for r in neg.itertuples(index=False):
            rows.append(
                {
                    "Record_ID": getattr(r, "Record_ID", np.nan),
                    "Period": getattr(r, "Period", np.nan),
                    "SKU_ID": r.SKU_ID,
                    "Region": r.Region,
                    "anomaly_type": "negative_units",
                    "severity": "high",
                    "value": r.Units_Sold,
                    "explanation": "Units_Sold < 0 after parsing. Treated as data error; excluded from demand reconstruction.",
                    "recommended_treatment": "Do not use as observed demand. Leave missing rather than clip unless revenue implies a valid quantity.",
                }
            )

    work = panel.dropna(subset=["Units_Sold_Observed"]).copy()
    work["sku_med"] = work.groupby("SKU_ID")["Units_Sold_Observed"].transform("median")
    work["rel"] = np.where(work["sku_med"] > 0, work["Units_Sold_Observed"] / work["sku_med"], np.nan)
    grp = work.groupby(["SKU_ID", "Region"], group_keys=False)
    work["iqr_flag"] = grp["Units_Sold_Observed"].transform(lambda s: _iqr_flags(s).astype(int))
    work["mom"] = grp["Units_Sold_Observed"].transform(lambda s: s.pct_change())
    work["struct_flag"] = (work["mom"].abs() > 2.0) & (work["Units_Sold_Observed"] > work["sku_med"].clip(lower=1) * 2)

    flagged = work[(work["iqr_flag"] == 1) | work["struct_flag"]]
    for r in flagged.itertuples(index=False):
        atype = "iqr_outlier" if r.iqr_flag == 1 else "structural_break_candidate"
        rows.append(
            {
                "Record_ID": np.nan,
                "Period": r.Period,
                "SKU_ID": r.SKU_ID,
                "Region": r.Region,
                "anomaly_type": atype,
                "severity": "medium",
                "value": r.Units_Sold_Observed,
                "explanation": "Series-level robust outlier or month-over-month change > 200%. May be a genuine disruption; not auto-removed.",
                "recommended_treatment": "Keep observed sales. Use reconstructed demand if a matching stockout exists; otherwise keep as-is.",
            }
        )

    # Isolation Forest on compact numeric features
    feat = work[["Units_Sold_Observed", "rel", "mom"]].replace([np.inf, -np.inf], np.nan)
    feat = feat.fillna(feat.median())
    if len(feat) >= 500:
        iso = IsolationForest(
            n_estimators=200,
            contamination=0.01,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        pred = iso.fit_predict(feat)
        iso_rows = work.loc[pred == -1]
        for r in iso_rows.itertuples(index=False):
            rows.append(
                {
                    "Record_ID": np.nan,
                    "Period": r.Period,
                    "SKU_ID": r.SKU_ID,
                    "Region": r.Region,
                    "anomaly_type": "isolation_forest",
                    "severity": "low",
                    "value": r.Units_Sold_Observed,
                    "explanation": "Multivariate IsolationForest score in the lower 1% of SKU-region monthly demand.",
                    "recommended_treatment": "Review; do not delete solely from IF score.",
                }
            )

    so = stockouts.copy()
    so = so[so["Lost_Units"].fillna(0) > 0]
    for r in so.itertuples(index=False):
        rows.append(
            {
                "Record_ID": np.nan,
                "Period": r.Period,
                "SKU_ID": r.SKU_ID,
                "Region": r.Region,
                "anomaly_type": "stockout",
                "severity": "high",
                "value": r.Lost_Units,
                "explanation": f"Stockout reason={getattr(r, 'Reason', '')}; lost_units recorded in inventory_stockouts.",
                "recommended_treatment": "Add documented Lost_Units to unconstrained demand. Do not invent extra lost sales.",
            }
        )

    # Customer ordering: months with all-zero for a previously active customer-SKU-region
    if {"Customer_Name", "Units_Sold", "Period"} <= set(train.columns):
        ct = (
            train.dropna(subset=["Period", "SKU_ID", "Region", "Customer_Name"])
            .groupby(["Customer_Name", "SKU_ID", "Region", "Period"], as_index=False)["Units_Sold"]
            .sum()
        )
        ct = ct.sort_values(["Customer_Name", "SKU_ID", "Region", "Period"])
        ct["prev"] = ct.groupby(["Customer_Name", "SKU_ID", "Region"])["Units_Sold"].shift(1)
        drop = (ct["Units_Sold"] == 0) & (ct["prev"] > 20)
        sample_drop = ct.loc[drop].head(5000)
        for r in sample_drop.itertuples(index=False):
            rows.append(
                {
                    "Record_ID": np.nan,
                    "Period": r.Period,
                    "SKU_ID": r.SKU_ID,
                    "Region": r.Region,
                    "anomaly_type": "customer_order_drop_to_zero",
                    "severity": "medium",
                    "value": 0,
                    "explanation": "Customer-SKU-region went from >20 units to 0. Could be genuine no-order (Units_Sold=0) or missing record.",
                    "recommended_treatment": "Keep zero as no order. Do not impute demand unless a stockout row exists.",
                }
            )

    register = pd.DataFrame(rows)
    if len(register):
        register = register.drop_duplicates(subset=["Period", "SKU_ID", "Region", "anomaly_type", "value"])
    from src.config import OUTPUT_DIR

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    register.to_csv(ANOMALY_REGISTER, index=False)
    return register
