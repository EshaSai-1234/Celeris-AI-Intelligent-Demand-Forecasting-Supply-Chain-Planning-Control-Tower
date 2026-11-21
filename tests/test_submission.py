import pandas as pd
import pytest

from src.config import SUBMISSION_COLUMNS
from src.submission import validate_submission


def test_validate_accepts_aligned_frame(tmp_path, monkeypatch):
    test = pd.DataFrame(
        {
            "Date": ["2026-11", "2026-12"],
            "SKU_ID": ["BSS-001", "BSS-001"],
            "Region": ["North America", "North America"],
        }
    )
    sub = test.copy()
    sub["Units_Sold_Forecast"] = [1.0, 2.0]
    sample = pd.DataFrame(columns=SUBMISSION_COLUMNS)
    sample_path = tmp_path / "sample_submission.csv"
    sample.to_csv(sample_path, index=False)
    monkeypatch.setattr("src.submission.SAMPLE_SUBMISSION_CSV", sample_path)
    validate_submission(sub[SUBMISSION_COLUMNS], test)


def test_validate_rejects_missing(tmp_path, monkeypatch):
    test = pd.DataFrame({"Date": ["2026-11"], "SKU_ID": ["BSS-001"], "Region": ["North America"]})
    sub = pd.DataFrame(
        {"Date": ["2026-11"], "SKU_ID": ["BSS-001"], "Region": ["North America"], "Units_Sold_Forecast": [float("nan")]}
    )
    sample_path = tmp_path / "sample_submission.csv"
    pd.DataFrame(columns=SUBMISSION_COLUMNS).to_csv(sample_path, index=False)
    monkeypatch.setattr("src.submission.SAMPLE_SUBMISSION_CSV", sample_path)
    with pytest.raises(ValueError):
        validate_submission(sub, test)
