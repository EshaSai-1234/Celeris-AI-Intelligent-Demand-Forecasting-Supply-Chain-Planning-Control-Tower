"""Baselines, Croston/SBA, global LightGBM, quantile models and recursive forecast."""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import LGBM_PARAMS, MODEL_DIR, QUANTILE_ALPHAS, RANDOM_STATE
from src.features import add_calendar_lifecycle, add_lags_and_rolls, add_macro_lags, event_flags, feature_columns
from src.utils import add_months


CAT_COLS = [
    "SKU_ID",
    "Region",
    "Product_Family",
    "Business_Unit",
    "Customer_Name",
    "Sales_Channel",
    "Size_Class",
    "Powertrain_Application",
    "Lifecycle_Status",
]


def seasonal_naive(history: pd.Series, horizon: int) -> np.ndarray:
    vals = history.dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return np.zeros(horizon)
    if len(vals) >= 12:
        season = vals[-12:]
        return np.resize(season, horizon)
    return np.full(horizon, vals[-1])


def moving_average(history: pd.Series, horizon: int, window: int = 3) -> np.ndarray:
    vals = history.dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return np.zeros(horizon)
    mu = vals[-window:].mean() if len(vals) else 0.0
    return np.full(horizon, mu)


def croston_sba(history: pd.Series, horizon: int, alpha: float = 0.1) -> np.ndarray:
    """Syntetos-Boylan approximation for intermittent demand. Returns a level forecast."""
    y = history.fillna(0).to_numpy(dtype=float)
    if len(y) == 0:
        return np.zeros(horizon)
    z = 0.0
    p = 1.0
    q = 1.0
    first = True
    for val in y:
        if val > 0:
            if first:
                z = val
                p = q
                first = False
            else:
                z = alpha * val + (1 - alpha) * z
                p = alpha * q + (1 - alpha) * p
            q = 1.0
        else:
            q += 1.0
    if first:
        return np.zeros(horizon)
    level = (1 - alpha / 2.0) * (z / max(p, 1e-6))
    return np.full(horizon, max(level, 0.0))


def _categorize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in CAT_COLS:
        if c in out.columns:
            out[c] = out[c].astype("category")
    return out


def prepare_xy(df: pd.DataFrame, cols: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    work = df.dropna(subset=["target"]).copy()
    drop = {"Period", "Date", "Year", "Month", "Year_Resolved", "Month_Resolved", "Period_Source"}
    cols = cols or [c for c in feature_columns(work) if c not in drop]
    cols = [c for c in cols if c in work.columns]
    work = _categorize(work)
    for c in cols:
        if work[c].dtype == object:
            work[c] = work[c].astype("category")
    X = work[cols]
    y = work["target"].astype(float)
    return X, y, cols


@dataclass
class LGBMBundle:
    point: lgb.LGBMRegressor
    quantiles: dict
    columns: list[str]


def train_lightgbm(train_df: pd.DataFrame, valid_df: pd.DataFrame | None = None) -> LGBMBundle:
    X, y, cols = prepare_xy(train_df)
    Xv = yv = None
    if valid_df is not None and len(valid_df.dropna(subset=["target"])):
        Xv, yv, _ = prepare_xy(valid_df, cols)
        Xv = Xv.reindex(columns=cols)
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    fit_kw = {}
    if Xv is not None:
        fit_kw["eval_set"] = [(Xv, yv)]
        fit_kw["callbacks"] = [lgb.early_stopping(40, verbose=False)]
    model.fit(X, y, **fit_kw)

    quantiles = {}
    for a in QUANTILE_ALPHAS:
        params = dict(LGBM_PARAMS)
        params.update({"objective": "quantile", "alpha": a, "n_estimators": min(400, LGBM_PARAMS["n_estimators"])})
        qm = lgb.LGBMRegressor(**params)
        qm.fit(X, y)
        quantiles[a] = qm

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "quantiles": quantiles, "columns": cols}, MODEL_DIR / "lgbm_bundle.joblib")
    return LGBMBundle(point=model, quantiles=quantiles, columns=cols)


def predict_lgbm(bundle: LGBMBundle, df: pd.DataFrame) -> np.ndarray:
    work = _categorize(df)
    for c in bundle.columns:
        if c in work.columns and work[c].dtype == object:
            work[c] = work[c].astype("category")
    X = work.reindex(columns=bundle.columns)
    pred = bundle.point.predict(X)
    return np.clip(pred, 0, None)


def predict_quantiles(bundle: LGBMBundle, df: pd.DataFrame) -> dict[str, np.ndarray]:
    work = _categorize(df)
    for c in bundle.columns:
        if c in work.columns and work[c].dtype == object:
            work[c] = work[c].astype("category")
    X = work.reindex(columns=bundle.columns)
    out = {}
    for a, m in bundle.quantiles.items():
        out[f"p{int(a*100):02d}"] = np.clip(m.predict(X), 0, None)
    return out


def baseline_panel_forecasts(
    history: pd.DataFrame,
    keys: pd.DataFrame,
    periods: list[str],
    target_col: str = "Demand_Unconstrained",
) -> pd.DataFrame:
    """Vectorized-enough per-key baselines for a list of future periods."""
    recs = []
    grouped = history.dropna(subset=["Period"]).sort_values("Period")
    for sku, region in keys[["SKU_ID", "Region"]].drop_duplicates().itertuples(index=False):
        h = grouped[(grouped["SKU_ID"] == sku) & (grouped["Region"] == region)][target_col]
        if h.empty:
            sn = ma = cr = np.zeros(len(periods))
        else:
            sn = seasonal_naive(h, len(periods))
            ma = moving_average(h, len(periods), 3)
            cr = croston_sba(h, len(periods))
        for i, p in enumerate(periods):
            recs.append(
                {
                    "Period": p,
                    "SKU_ID": sku,
                    "Region": region,
                    "pred_seasonal_naive": max(sn[i], 0),
                    "pred_moving_average": max(ma[i], 0),
                    "pred_croston_sba": max(cr[i], 0),
                }
            )
    return pd.DataFrame(recs)


def recursive_lgbm_forecast(
    bundle: LGBMBundle,
    history_panel: pd.DataFrame,
    future_keys: pd.DataFrame,
    future_periods: list[str],
    macro: pd.DataFrame,
    events: pd.DataFrame,
    sku: pd.DataFrame,
) -> pd.DataFrame:
    """Walk month by month, updating only demand lags so features stay causal."""
    hist = history_panel.copy()
    if "Demand_Unconstrained" not in hist.columns:
        hist["Demand_Unconstrained"] = hist.get("target")
    static_cols = [
        "SKU_ID",
        "Region",
        "Product_Family",
        "Business_Unit",
        "Customer_Name",
        "Sales_Channel",
        "Size_Class",
        "Powertrain_Application",
        "Launch_Period",
        "EOL_Period",
        "Replaces_SKU",
        "Lifecycle_Status",
    ]
    hist_static = hist[[c for c in static_cols if c in hist.columns]].drop_duplicates(["SKU_ID", "Region"])
    fut_static = future_keys[[c for c in static_cols if c in future_keys.columns]].drop_duplicates(["SKU_ID", "Region"])
    static = pd.concat([hist_static, fut_static], ignore_index=True).drop_duplicates(["SKU_ID", "Region"], keep="last")

    grid = static.assign(_k=1).merge(pd.DataFrame({"Period": future_periods, "_k": 1}), on="_k").drop(columns="_k")
    grid = add_calendar_lifecycle(grid)
    grid = event_flags(grid, events)
    grid = add_macro_lags(grid, macro)

    demand = {}
    hp = hist.dropna(subset=["Period"])
    for (sku_id, region), g in hp.groupby(["SKU_ID", "Region"]):
        g = g.sort_values("Period")
        demand[(sku_id, region)] = dict(zip(g["Period"].astype(str), g["Demand_Unconstrained"].astype(float)))

    fam_hist = (
        hp.groupby(["Product_Family", "Region", "Period"], dropna=False)["Demand_Unconstrained"]
        .mean()
        .to_dict()
    )

    preds = []
    for p in future_periods:
        cur = grid[grid["Period"] == p].copy()
        lag_vals = {lag: [] for lag in (1, 2, 3, 6, 12)}
        rolls = {f"roll_mean_{w}": [] for w in (3, 6, 12)}
        rolls.update({f"roll_std_{w}": [] for w in (3, 6, 12)})
        zeros = []
        fam_lag = []
        for sku_id, region, fam in zip(cur["SKU_ID"], cur["Region"], cur["Product_Family"]):
            series = demand.get((sku_id, region), {})
            hist_periods = sorted(series)
            vals = [series[k] for k in hist_periods]
            for lag in (1, 2, 3, 6, 12):
                lag_p = add_months(p, -lag)
                lag_vals[lag].append(series.get(lag_p, np.nan))
            tail = pd.Series(vals[-12:], dtype=float)
            for w in (3, 6, 12):
                window = tail.iloc[-w:] if len(tail) else tail
                rolls[f"roll_mean_{w}"].append(float(window.mean()) if len(window) else np.nan)
                rolls[f"roll_std_{w}"].append(float(window.std()) if len(window) > 1 else np.nan)
            zeros.append(float((tail == 0).mean()) if len(tail) else np.nan)
            fam_lag.append(fam_hist.get((fam, region, add_months(p, -1)), np.nan))
        for lag in (1, 2, 3, 6, 12):
            cur[f"Demand_Unconstrained_lag{lag}"] = lag_vals[lag]
        for k, v in rolls.items():
            cur[k] = v
        cur["zero_share_12"] = zeros
        cur["family_region_mean_lag1"] = fam_lag
        cur["target"] = np.nan
        yhat = predict_lgbm(bundle, cur)
        q = predict_quantiles(bundle, cur)
        cur["pred_lgbm"] = yhat
        for k, v in q.items():
            cur[k] = v
        pre = cur["is_pre_launch"].fillna(0).eq(1)
        past = cur["is_past_eol"].fillna(0).eq(1)
        cur.loc[pre | past, ["pred_lgbm", "p10", "p50", "p90"]] = 0.0
        for sku_id, region, fam, val in zip(cur["SKU_ID"], cur["Region"], cur["Product_Family"], cur["pred_lgbm"]):
            demand.setdefault((sku_id, region), {})[p] = float(val)
            fam_hist[(fam, region, p)] = float(val)
        preds.append(cur)
    out = pd.concat(preds, ignore_index=True)
    keep = [
        "Period",
        "SKU_ID",
        "Region",
        "Product_Family",
        "Customer_Name",
        "Sales_Channel",
        "pred_lgbm",
        "p10",
        "p50",
        "p90",
        "is_pre_launch",
        "is_past_eol",
        "Lifecycle_Status",
    ]
    return out[[c for c in keep if c in out.columns]]
