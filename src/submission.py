"""Build and validate submission.csv against sample schema and test row order."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import SAMPLE_SUBMISSION_CSV, SUBMISSION_COLUMNS, SUBMISSION_CSV, TEST_CSV


def build_submission(test: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    raw_test = pd.read_csv(TEST_CSV)
    # Preserve original Date / SKU_ID / Region strings from test.csv, not cleaned versions.
    keys = raw_test[["Date", "SKU_ID", "Region"]].copy()
    keys["submission_row"] = np.arange(len(keys), dtype=int)

    f = forecast.copy()
    f = f.rename(columns={"Period": "Date"})
    value_col = "pred_reconciled" if "pred_reconciled" in f.columns else "pred_lgbm"
    f["Units_Sold_Forecast"] = np.clip(f[value_col].astype(float), 0, None)
    # Merge on cleaned Date-SKU-Region equals original for test (already canonical).
    merged = keys.merge(
        f[["Date", "SKU_ID", "Region", "Units_Sold_Forecast"]],
        on=["Date", "SKU_ID", "Region"],
        how="left",
    )
    if merged["Units_Sold_Forecast"].isna().any():
        # Fallback: cleaned test row map
        t = test[["submission_row", "Period", "SKU_ID", "Region"]].copy()
        t = t.merge(
            forecast.rename(columns={"Period": "Period"})[
                ["Period", "SKU_ID", "Region", value_col]
            ],
            left_on=["Period", "SKU_ID", "Region"],
            right_on=["Period", "SKU_ID", "Region"],
            how="left",
        )
        t["Units_Sold_Forecast"] = np.clip(t[value_col].astype(float), 0, None)
        merged.loc[merged["Units_Sold_Forecast"].isna(), "Units_Sold_Forecast"] = t.set_index(
            "submission_row"
        ).loc[merged.loc[merged["Units_Sold_Forecast"].isna(), "submission_row"], "Units_Sold_Forecast"].to_numpy()

    merged["Units_Sold_Forecast"] = merged["Units_Sold_Forecast"].fillna(0.0)
    merged["Units_Sold_Forecast"] = np.where(
        np.isfinite(merged["Units_Sold_Forecast"]), merged["Units_Sold_Forecast"], 0.0
    )
    merged["Units_Sold_Forecast"] = merged["Units_Sold_Forecast"].clip(lower=0)
    out = merged.sort_values("submission_row")[SUBMISSION_COLUMNS]
    validate_submission(out, raw_test)
    out.to_csv(SUBMISSION_CSV, index=False)
    out.to_csv(OUTPUT_DIR_SUBMISSION_ROOT(), index=False)
    return out


def OUTPUT_DIR_SUBMISSION_ROOT():
    from src.config import ROOT

    return ROOT / "submission.csv"


def validate_submission(sub: pd.DataFrame, raw_test: pd.DataFrame) -> None:
    sample = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    if list(sub.columns) != list(sample.columns):
        raise ValueError(f"Submission columns {list(sub.columns)} != sample {list(sample.columns)}")
    if len(sub) != len(raw_test):
        raise ValueError(f"Submission has {len(sub)} rows, test has {len(raw_test)}")
    if not sub["Date"].astype(str).equals(raw_test["Date"].astype(str)):
        raise ValueError("Date column order/values do not match test.csv")
    if not sub["SKU_ID"].astype(str).equals(raw_test["SKU_ID"].astype(str)):
        raise ValueError("SKU_ID column order/values do not match test.csv")
    if not sub["Region"].astype(str).equals(raw_test["Region"].astype(str)):
        raise ValueError("Region column order/values do not match test.csv")
    vals = sub["Units_Sold_Forecast"]
    if vals.isna().any():
        raise ValueError("Missing predictions")
    if not np.isfinite(vals.astype(float)).all():
        raise ValueError("Non-finite predictions")
    if (vals.astype(float) < 0).any():
        raise ValueError("Negative predictions")
