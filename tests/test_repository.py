from datetime import datetime
from api.db.repository import get_error_flags
from api.schemas.transaction_schema import Transaction


def make_transaction(errors):
    return Transaction(
        id=1,
        date=datetime(2026, 1, 1),
        client_id=1,
        card_id=1,
        amount=10.0,
        use_chip="Online Transaction",
        merchant_id=1,
        mcc=1000,
        errors=errors,
    )


def test_no_errors():
    flags = get_error_flags(make_transaction(None))
    assert flags == {"has_bad_pin": 0, "has_bad_cvv": 0, "has_insufficient_balance": 0, "has_technical_glitch": 0}


def test_bad_pin_detected_case_insensitive():
    flags = get_error_flags(make_transaction("Bad PIN"))
    assert flags["has_bad_pin"] == 1
    assert flags["has_bad_cvv"] == 0


def test_multiple_errors_detected():
    flags = get_error_flags(make_transaction("Insufficient Balance,Technical Glitch"))
    assert flags["has_insufficient_balance"] == 1
    assert flags["has_technical_glitch"] == 1
    assert flags["has_bad_pin"] == 0
