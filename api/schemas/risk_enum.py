import enum


class RISK_LEVEL(enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class REVIEW_STATUS(enum.Enum):
    APPROVED = "approved"
    PENDING = "pending"
    REJECT = "reject"