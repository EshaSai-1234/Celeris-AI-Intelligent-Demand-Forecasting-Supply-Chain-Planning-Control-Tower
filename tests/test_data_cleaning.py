import pytest
import numpy as np
import pandas as pd
from src.data_cleaning import parse_date_robust, clean_units_sold

def test_parse_date_robust():
    assert parse_date_robust("2019-01") == "2019-01"
    assert parse_date_robust("2019/05") == "2019-05"
    assert parse_date_robust("03/2020") == "2020-03"
    assert parse_date_robust("Jan 2019") == "2019-01"
    assert parse_date_robust("Feb-2022") == "2022-02"
    assert parse_date_robust("2021-08-15") == "2021-08"

def test_clean_units_sold():
    assert clean_units_sold("9,600") == 9600.0
    assert clean_units_sold("-") == 0.0
    assert clean_units_sold("0") == 0.0
    assert clean_units_sold("150") == 150.0
    assert np.isnan(clean_units_sold("invalid"))
