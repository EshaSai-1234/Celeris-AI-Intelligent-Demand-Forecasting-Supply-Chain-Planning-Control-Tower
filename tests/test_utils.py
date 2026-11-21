import numpy as np
import pandas as pd

from src.utils import (
    normalize_channel,
    normalize_customer,
    normalize_region,
    normalize_sku_id,
    parse_units,
    parse_year_month,
    resolve_period_from_row,
)


def test_parse_year_month_formats():
    assert parse_year_month("2019-01") == (2019, 1)
    assert parse_year_month("2019/01") == (2019, 1)
    assert parse_year_month("01/2019") == (2019, 1)
    assert parse_year_month("01-2019") == (2019, 1)
    assert parse_year_month("Jan 2019") == (2019, 1)
    assert parse_year_month("Jan-2019") == (2019, 1)
    assert parse_year_month("2019-01-15") == (2019, 1)


def test_trust_date_over_year_month():
    period, src = resolve_period_from_row("2020-03", 2019, 1)
    assert period == "2020-03"
    assert src == "date_over_year_month"


def test_implausible_date_falls_back():
    period, src = resolve_period_from_row("2062-02", 2026, 2)
    assert period == "2026-02"
    assert "fallback" in src


def test_normalize_dims():
    assert normalize_region("LATAM") == "Latin America"
    assert normalize_region("APAC") == "Asia Pacific"
    assert normalize_channel("OEM Direct") == "Direct OEM"
    assert normalize_customer("APEX TIER ONE") == "Apex Tier One"
    assert normalize_sku_id(" bss-1 ") == "BSS-001"
    assert normalize_sku_id("PMP_023") == "PMP-023"


def test_parse_units():
    assert parse_units("4,400") == 4400
    assert np.isnan(parse_units("-"))
    assert parse_units("0") == 0.0
