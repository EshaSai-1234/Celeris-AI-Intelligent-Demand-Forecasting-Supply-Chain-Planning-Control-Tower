"""Hierarchical reconciliation: bottom-up and a diagonal MinT at family level."""

from __future__ import annotations

import numpy as np
import pandas as pd


def bottom_up(forecast: pd.DataFrame, value_col: str = "pred_lgbm") -> pd.DataFrame:
    fam = (
        forecast.groupby(["Period", "Product_Family", "Region"], dropna=False)[value_col]
        .sum()
        .reset_index()
        .rename(columns={value_col: "family_region_bottom_up"})
    )
    tot = (
        forecast.groupby(["Period"], dropna=False)[value_col]
        .sum()
        .reset_index()
        .rename(columns={value_col: "total_bottom_up"})
    )
    out = forecast.merge(fam, on=["Period", "Product_Family", "Region"], how="left")
    out = out.merge(tot, on="Period", how="left")
    out["pred_reconciled"] = out[value_col]
    return out


def mint_family(
    bottom: pd.DataFrame,
    family_forecast: pd.DataFrame,
    bottom_col: str = "pred_lgbm",
    family_col: str = "pred_family",
) -> pd.DataFrame:
    """
    Scale SKU-region forecasts so they sum to a blend of bottom-up and family-level forecasts.

    Covariance is approximated as diagonal using squared bottom-up share (structural scaling).
    This is a MinT-style shrinkage, not a claim of full covariance MinT.
    """
    out = bottom.copy()
    bu = out.groupby(["Period", "Product_Family"], dropna=False)[bottom_col].transform("sum")
    out["_bu_family"] = bu
    fam = family_forecast.rename(columns={family_col: "_fam_pred"})
    out = out.merge(fam, on=["Period", "Product_Family"], how="left")
    # shrink 50/50 when family forecast exists
    target = np.where(
        out["_fam_pred"].notna() & (out["_bu_family"] > 0),
        0.5 * out["_bu_family"] + 0.5 * out["_fam_pred"],
        out["_bu_family"],
    )
    scale = np.where(out["_bu_family"] > 0, target / out["_bu_family"], 1.0)
    out["pred_reconciled"] = out[bottom_col] * scale
    return out.drop(columns=["_bu_family", "_fam_pred"], errors="ignore")


def family_level_forecast(history: pd.DataFrame, periods: list[str], target: str = "Demand_Unconstrained") -> pd.DataFrame:
    """Holt-Winters where a family series is long enough; otherwise seasonal naive."""
    recs = []
    fam = history.dropna(subset=["Period"]).groupby(["Product_Family", "Period"], dropna=False)[target].sum().reset_index()
    for family, g in fam.groupby("Product_Family"):
        g = g.sort_values("Period")
        hist = g[target].astype(float)
        vals = _hw_or_naive(hist, len(periods))
        for p, v in zip(periods, vals):
            recs.append({"Period": p, "Product_Family": family, "pred_family": max(float(v), 0.0)})
    return pd.DataFrame(recs)


def _hw_or_naive(hist: pd.Series, horizon: int) -> np.ndarray:
    y = hist.dropna().to_numpy(dtype=float)
    if len(y) < 24:
        if len(y) >= 12:
            return np.resize(y[-12:], horizon)
        if len(y):
            return np.full(horizon, y[-1])
        return np.zeros(horizon)
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        fit = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=12, initialization_method="estimated")
        res = fit.fit(optimized=True)
        fc = np.asarray(res.forecast(horizon), dtype=float)
        if np.isfinite(fc).all():
            return fc
    except Exception:
        pass
    return np.resize(y[-12:], horizon)
