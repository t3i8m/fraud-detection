from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel
from risk_enum import RISK_LEVEL


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
