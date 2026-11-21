"""Shared parsing, period helpers and category normalization."""

from __future__ import annotations

import json
import re
from datetime import datetime
from difflib import get_close_matches
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import (
    CANONICAL_CHANNELS,
    CANONICAL_CUSTOMERS,
    CANONICAL_REGIONS,
    CHANNEL_ALIASES,
    REGION_ALIASES,
    VALID_YEAR_MAX,
    VALID_YEAR_MIN,
)

MONTH_NAME_TO_NUM = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

SKU_RE = re.compile(r"^([A-Z]{3})[\s\-_]*([0-9]{1,3})$")


def parse_year_month(value: object) -> tuple[float, float]:
    """Parse a date-like value to (year, month). Returns (nan, nan) if unknown."""
    if pd.isna(value):
        return (np.nan, np.nan)
    if isinstance(value, (datetime, pd.Timestamp)):
        return (int(value.year), int(value.month))
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "nat"}:
        return (np.nan, np.nan)

    m = re.match(r"^(\d{4})[-/](\d{1,2})(?:[-/]\d{1,2})?$", s)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12:
            return (year, month)

    m = re.match(r"^(\d{1,2})[-/](\d{4})$", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if 1 <= a <= 12:
            return (b, a)

    m = re.match(r"^([A-Za-z]{3,9})[\s\-/]+(\d{4})$", s)
    if m and m.group(1).lower() in MONTH_NAME_TO_NUM:
        return (int(m.group(2)), MONTH_NAME_TO_NUM[m.group(1).lower()])

    m = re.match(r"^(\d{4})[\s\-/]+([A-Za-z]{3,9})$", s)
    if m and m.group(2).lower() in MONTH_NAME_TO_NUM:
        return (int(m.group(1)), MONTH_NAME_TO_NUM[m.group(2).lower()])

    ts = pd.to_datetime(s, errors="coerce")
    if pd.notna(ts):
        return (int(ts.year), int(ts.month))
    return (np.nan, np.nan)


def to_period_str(year: object, month: object) -> str | float:
    if pd.isna(year) or pd.isna(month):
        return np.nan
    year_i, month_i = int(year), int(month)
    if month_i < 1 or month_i > 12:
        return np.nan
    return f"{year_i:04d}-{month_i:02d}"


def period_to_timestamp(period: str) -> pd.Timestamp:
    return pd.Timestamp(f"{period}-01")


def add_months(period: str, n: int) -> str:
    ts = period_to_timestamp(period) + pd.DateOffset(months=n)
    return f"{ts.year:04d}-{ts.month:02d}"


def period_range(start: str, end: str) -> list[str]:
    out: list[str] = []
    cur = start
    while cur <= end:
        out.append(cur)
        cur = add_months(cur, 1)
    return out


def months_between(start: str, end: str) -> int:
    a = period_to_timestamp(start)
    b = period_to_timestamp(end)
    return int((b.year - a.year) * 12 + (b.month - a.month))


def resolve_period_from_row(date_val: object, year_val: object, month_val: object) -> tuple[str | float, str]:
    """Trust Date when it is a valid business period. Fall back to Year/Month for implausible Date years."""
    y_d, m_d = parse_year_month(date_val)
    y_col = pd.to_numeric(year_val, errors="coerce")
    m_col = pd.to_numeric(month_val, errors="coerce")

    date_ok = (
        pd.notna(y_d)
        and pd.notna(m_d)
        and VALID_YEAR_MIN <= int(y_d) <= VALID_YEAR_MAX
        and 1 <= int(m_d) <= 12
    )
    ym_ok = pd.notna(y_col) and pd.notna(m_col) and 1 <= int(m_col) <= 12

    if date_ok:
        period = to_period_str(y_d, m_d)
        note = "date"
        if ym_ok and (int(y_d) != int(y_col) or int(m_d) != int(m_col)):
            note = "date_over_year_month"
        return period, note

    if ym_ok:
        y_i = int(y_col)
        if VALID_YEAR_MIN <= y_i <= VALID_YEAR_MAX:
            return to_period_str(y_i, int(m_col)), "year_month_fallback_implausible_date"
        return to_period_str(y_i, int(m_col)), "year_month_out_of_range"

    return np.nan, "unparseable"


def normalize_region(value: object) -> object:
    if pd.isna(value):
        return np.nan
    key = re.sub(r"\s+", " ", str(value).strip().lower())
    if key in REGION_ALIASES:
        return REGION_ALIASES[key]
    for canon in CANONICAL_REGIONS:
        if key == canon.lower():
            return canon
    return str(value).strip()


def normalize_channel(value: object) -> object:
    if pd.isna(value):
        return np.nan
    key = re.sub(r"\s+", " ", str(value).strip().lower())
    if key in CHANNEL_ALIASES:
        return CHANNEL_ALIASES[key]
    for canon in CANONICAL_CHANNELS:
        if key == canon.lower():
            return canon
    return str(value).strip()


def _customer_key(value: str) -> str:
    s = re.sub(r"\s+", " ", value.strip())
    s = re.sub(r"\s+inc\.?$", "", s, flags=re.I)
    return s.lower()


_CUSTOMER_LOOKUP = {_customer_key(c): c for c in CANONICAL_CUSTOMERS}


def normalize_customer(value: object) -> object:
    if pd.isna(value):
        return np.nan
    raw = str(value).strip()
    key = _customer_key(raw)
    if key in _CUSTOMER_LOOKUP:
        return _CUSTOMER_LOOKUP[key]
    matches = get_close_matches(key, list(_CUSTOMER_LOOKUP.keys()), n=1, cutoff=0.78)
    if matches:
        return _CUSTOMER_LOOKUP[matches[0]]
    return raw


def normalize_sku_id(value: object, valid_skus: Iterable[str] | None = None) -> object:
    if pd.isna(value):
        return np.nan
    s = str(value).strip().upper()
    m = SKU_RE.match(s)
    if not m:
        compact = re.sub(r"[^A-Z0-9]", "", s)
        m = re.match(r"^([A-Z]{3})([0-9]{1,3})$", compact)
        if not m:
            return str(value).strip()
    sku = f"{m.group(1)}-{int(m.group(2)):03d}"
    if valid_skus is not None and sku not in set(valid_skus):
        return sku
    return sku


def parse_units(value: object) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    s = str(value).strip().replace(",", "")
    if s in {"", "-", "NA", "N/A", "null", "None"}:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def json_dump(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
