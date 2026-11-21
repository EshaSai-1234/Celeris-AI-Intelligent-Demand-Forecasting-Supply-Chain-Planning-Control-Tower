"""SHAP explanations for the global LightGBM model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import OUTPUT_DIR, SHAP_GLOBAL, SHAP_SKU
from src.models import CAT_COLS, LGBMBundle, _categorize


def compute_shap(bundle: LGBMBundle, sample: pd.DataFrame, max_rows: int = 4000) -> tuple[pd.DataFrame, pd.DataFrame]:
    import shap

    work = sample.dropna(subset=["target"] if "target" in sample.columns else []).copy()
    if len(work) > max_rows:
        work = work.sample(max_rows, random_state=42)
    X = _categorize(work).reindex(columns=bundle.columns)
    explainer = shap.TreeExplainer(bundle.point)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[0]
    abs_mean = np.abs(sv).mean(axis=0)
    global_imp = (
        pd.DataFrame({"feature": bundle.columns, "mean_abs_shap": abs_mean})
        .sort_values("mean_abs_shap", ascending=False)
    )
    global_imp.to_csv(SHAP_GLOBAL, index=False)

    # SKU-level: mean |SHAP| of lag-1 and month features plus prediction residual proxy
    sku_rows = []
    if "SKU_ID" in work.columns:
        work = work.reset_index(drop=True)
        pred = bundle.point.predict(X)
        for sku, idx in work.groupby("SKU_ID").groups.items():
            idx = list(idx)
            if not idx:
                continue
            local = np.abs(sv[idx]).mean(axis=0)
            top_i = int(np.argmax(local))
            sku_rows.append(
                {
                    "SKU_ID": sku,
                    "n_rows_explained": len(idx),
                    "mean_prediction": float(np.mean(pred[idx])),
                    "top_feature": bundle.columns[top_i],
                    "top_feature_mean_abs_shap": float(local[top_i]),
                }
            )
    sku_df = pd.DataFrame(sku_rows)
    sku_df.to_csv(SHAP_SKU, index=False)
    return global_imp, sku_df
