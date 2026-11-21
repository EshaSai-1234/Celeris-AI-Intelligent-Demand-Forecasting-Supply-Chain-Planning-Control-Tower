import pytest
import pandas as pd
import numpy as np

def test_unconstrained_demand_calculation():
    # Simulated row with stockout
    units_sold = 40
    lost_units = 15
    unconstrained = max(units_sold, units_sold + lost_units)
    assert unconstrained == 55
    assert unconstrained >= units_sold

def test_non_stockout_demand():
    units_sold = 30
    lost_units = 0
    unconstrained = max(units_sold, units_sold + lost_units)
    assert unconstrained == 30
