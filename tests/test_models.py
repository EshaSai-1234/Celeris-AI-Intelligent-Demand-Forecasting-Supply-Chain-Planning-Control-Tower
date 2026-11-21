import pytest
import pandas as pd
import numpy as np
from src.models.baselines import SeasonalNaiveForecaster, MovingAverageForecaster, CrostonSBAForecaster
from src.models.lifecycle_handler import LifecycleHandler

def test_baselines():
    # Synthetic series
    df = pd.DataFrame({
        'SKU_ID': ['SKU-1', 'SKU-1', 'SKU-1', 'SKU-1'],
        'Region': ['Europe', 'Europe', 'Europe', 'Europe'],
        'Date': ['2026-01', '2026-02', '2026-03', '2026-04'],
        'Month': [1, 2, 3, 4],
        'Unconstrained_Demand': [10, 20, 30, 40]
    })
    
    test_df = pd.DataFrame({
        'SKU_ID': ['SKU-1'],
        'Region': ['Europe'],
        'Date': ['2026-05'],
        'Month': [5]
    })
    
    # Test Moving Average
    ma = MovingAverageForecaster(window=2).fit(df)
    preds = ma.predict(test_df)
    assert len(preds) == 1
    assert preds[0] == 35  # mean of [30, 40]
    
    # Test Seasonal Naive
    sn = SeasonalNaiveForecaster().fit(df)
    preds_sn = sn.predict(test_df)
    assert len(preds_sn) == 1
    assert preds_sn[0] >= 0

def test_croston_sba():
    df_intermittent = pd.DataFrame({
        'SKU_ID': ['SKU-2', 'SKU-2', 'SKU-2', 'SKU-2', 'SKU-2'],
        'Region': ['India', 'India', 'India', 'India', 'India'],
        'Date': ['2026-01', '2026-02', '2026-03', '2026-04', '2026-05'],
        'Unconstrained_Demand': [0, 50, 0, 0, 60]
    })
    test_df = pd.DataFrame({
        'SKU_ID': ['SKU-2'],
        'Region': ['India'],
        'Date': ['2026-06'],
        'Month': [6]
    })
    croston = CrostonSBAForecaster(alpha=0.1).fit(df_intermittent)
    preds = croston.predict(test_df)
    assert len(preds) == 1
    assert preds[0] > 0
