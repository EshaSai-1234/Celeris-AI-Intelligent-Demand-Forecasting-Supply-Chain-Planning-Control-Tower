import os
import pytest
import pandas as pd
from src.config import SUBMISSION_FILE, SAMPLE_SUBMISSION_FILE, TEST_FILE

def test_submission_exists_and_format():
    if not os.path.exists(SUBMISSION_FILE):
        pytest.skip("Submission file not yet generated.")
        
    sub = pd.read_csv(SUBMISSION_FILE)
    test = pd.read_csv(TEST_FILE)
    sample_sub = pd.read_csv(SAMPLE_SUBMISSION_FILE)
    
    # 1. Exact row count match
    assert len(sub) == len(test), f"Row count mismatch: expected {len(test)}, got {len(sub)}"
    
    # 2. Exact column schema match
    assert list(sub.columns) == list(sample_sub.columns), f"Columns mismatch: expected {sample_sub.columns}, got {sub.columns}"
    
    # 3. No nulls or missing values
    assert sub['Units_Sold_Forecast'].isna().sum() == 0, "Submission contains null forecasts"
    
    # 4. Non-negative integer predictions
    assert (sub['Units_Sold_Forecast'] < 0).sum() == 0, "Submission contains negative forecasts"
    assert (sub['Units_Sold_Forecast'] % 1 != 0).sum() == 0, "Forecasts must be integers"
