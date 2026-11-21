"""Chronological rolling-origin validation. No random splits."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import BACKTEST_REPORT, MODEL_COMPARISON
from src.models import croston_sba, moving_average, predict_lgbm, seasonal_naive, train_lightgbm
from src.utils import add_months, months_between, period_range


def mae(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs(y - p)))


def rmse(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def wape(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    denom = np.abs(y).sum()
    if denom == 0:
        return np.nan
    return float(np.abs(y - p).sum() / denom)


def bias(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(p - y))


def coverage(y, lo, hi) -> float:
    y = np.asarray(y, float)
    return float(((y >= np.asarray(lo)) & (y <= np.asarray(hi))).mean())


def rolling_origin_baselines(feat: pd.DataFrame, origins: list[str], horizons: list[int]) -> pd.DataFrame:
    labeled = feat.dropna(subset=["target", "Period"]).copy()
    grouped = [(k, g.sort_values("Period")) for k, g in labeled.groupby(["SKU_ID", "Region"])]
    rows = []
    for origin in origins:
        for h in horizons:
            months = period_range(add_months(origin, 1), add_months(origin, h))
            buckets = {
                "seasonal_naive": {"y": [], "p": []},
                "moving_average": {"y": [], "p": []},
                "croston_sba": {"y": [], "p": []},
            }
            for _, g in grouped:
                hist = g.loc[g["Period"] <= origin, "target"]
                actual = g.loc[g["Period"].isin(months)]
                if actual.empty:
                    continue
                n = len(months)
                fmap = {
                    "seasonal_naive": dict(zip(months, seasonal_naive(hist, n))),
                    "moving_average": dict(zip(months, moving_average(hist, n, 3))),
                    "croston_sba": dict(zip(months, croston_sba(hist, n))),
                }
                y = actual["target"].to_numpy(dtype=float)
                for name in buckets:
                    buckets[name]["y"].append(y)
                    buckets[name]["p"].append(actual["Period"].map(fmap[name]).to_numpy(dtype=float))
            for name, d in buckets.items():
                if not d["y"]:
                    continue
                y = np.concatenate(d["y"])
                p = np.concatenate(d["p"])
                rows.append(
                    {
                        "origin": origin,
                        "horizon_months": h,
                        "model": name,
                        "n": int(len(y)),
                        "MAE": mae(y, p),
                        "RMSE": rmse(y, p),
                        "WAPE": wape(y, p),
                        "Bias": bias(y, p),
                        "coverage_p10_p90": np.nan,
                    }
                )
    return pd.DataFrame(rows)


def rolling_origin_lgbm(feat: pd.DataFrame, origins: list[str], horizons: list[int]) -> pd.DataFrame:
    from src.models import _categorize

    rows = []
    for origin in origins:
        train_df = feat[(feat["Period"] <= origin) & feat["target"].notna()]
        if len(train_df) < 1000:
            continue
        max_h = max(horizons)
        test_df = feat[
            (feat["Period"] > origin)
            & (feat["Period"] <= add_months(origin, max_h))
            & feat["target"].notna()
        ]
        if test_df.empty:
            continue
        bundle = train_lightgbm(train_df, valid_df=None)
        pred = predict_lgbm(bundle, test_df)
        tmp = test_df[["Period", "target"]].copy()
        tmp["pred"] = pred
        tmp["h"] = tmp["Period"].map(lambda p: months_between(origin, p))
        X = _categorize(test_df).reindex(columns=bundle.columns)
        p10 = np.clip(bundle.quantiles[0.1].predict(X), 0, None)
        p90 = np.clip(bundle.quantiles[0.9].predict(X), 0, None)
        tmp["p10"] = p10
        tmp["p90"] = p90
        for h in horizons:
            part = tmp[tmp["h"] <= h]
            if part.empty:
                continue
            rows.append(
                {
                    "origin": origin,
                    "horizon_months": h,
                    "model": "lightgbm_global",
                    "n": int(len(part)),
                    "MAE": mae(part["target"], part["pred"]),
                    "RMSE": rmse(part["target"], part["pred"]),
                    "WAPE": wape(part["target"], part["pred"]),
                    "Bias": bias(part["target"], part["pred"]),
                    "coverage_p10_p90": coverage(part["target"], part["p10"], part["p90"]),
                }
            )
    return pd.DataFrame(rows)


def summarize_backtest(base_rows: pd.DataFrame, lgbm_rows: pd.DataFrame) -> pd.DataFrame:
    comparison = pd.concat([base_rows, lgbm_rows], ignore_index=True, sort=False)
    comparison.to_csv(MODEL_COMPARISON, index=False)
    comparison.to_csv(BACKTEST_REPORT, index=False)
    return comparison
