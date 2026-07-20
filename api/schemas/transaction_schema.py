from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel
from api.schemas.risk_enum import RISK_LEVEL


class Transaction(BaseModel):
    id: int
    date: datetime
    client_id: int
    card_id: int
    amount: float
    use_chip: Literal["Online Transaction", "Swipe Transaction", "Chip Transaction"]
    merchant_id: int
    merchant_city: Optional[str] = None
    merchant_state: Optional[str] = None
    zip: Optional[str] = None
    mcc: int
    errors: Optional[str] = None


class TransactionPredicted(Transaction):
    fraud_probability: float
    risk_level: RISK_LEVEL
    time_since_last_trx: float
    online_history_ratio: float
    is_new_merchant: int
    is_new_mcc: int
    user_amount_z_score: float
    trx_count_1h: int
    trx_amount_1h: float
    has_bad_pin: int
    has_insufficient_balance: int
    has_technical_glitch: int
    bad_pin_count_1h: int
    bad_cvv_count_1h: int
    insufficient_balance_count_1h: int
    tech_glitch_count_1h: int
