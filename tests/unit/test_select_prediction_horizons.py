from datetime import datetime, timedelta

import pytest

from src.langgraph_workflow.utils import select_prediction_horizons


@pytest.mark.unit
def test_select_prediction_horizons_low_risk_shortens_range():
    target = datetime.now() + timedelta(days=200)
    horizons = select_prediction_horizons(target, "low")
    assert horizons == sorted(set([1, 7, 30, 200]))


@pytest.mark.unit
def test_select_prediction_horizons_high_risk_includes_long_term():
    target = datetime.now() + timedelta(days=500)
    horizons = select_prediction_horizons(target, "high")
    assert 365 in horizons
    assert horizons[-1] == 500


@pytest.mark.unit
def test_select_prediction_horizons_requires_valid_risk():
    target = datetime.now() + timedelta(days=30)
    with pytest.raises(ValueError):
        select_prediction_horizons(target, "unknown")
