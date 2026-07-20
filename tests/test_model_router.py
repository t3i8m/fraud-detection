from api.routers.model_router import get_risk_level
from api.schemas.risk_enum import RISK_LEVEL


def test_low_risk_below_threshold():
    assert get_risk_level(0.0) == RISK_LEVEL.LOW
    assert get_risk_level(0.204) == RISK_LEVEL.LOW


def test_medium_risk_between_thresholds():
    assert get_risk_level(0.205) == RISK_LEVEL.MEDIUM
    assert get_risk_level(0.5) == RISK_LEVEL.MEDIUM
    assert get_risk_level(0.829) == RISK_LEVEL.MEDIUM


def test_high_risk_at_and_above_threshold():
    assert get_risk_level(0.830) == RISK_LEVEL.HIGH
    assert get_risk_level(1.0) == RISK_LEVEL.HIGH
