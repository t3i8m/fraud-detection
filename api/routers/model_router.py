from fastapi import APIRouter, Request
import pandas as pd
from api.config import HIGH_RISK_THRESHOLD, LOW_RISK_THRESHOLD
from api.db.repository import get_realtime_features, update_redis_cache, save_transaction
from api.schemas.risk_enum import RISK_LEVEL
from api.schemas.transaction_schema import Transaction, TransactionPredicted


router = APIRouter()


def get_risk_level(probability):
    if probability < LOW_RISK_THRESHOLD:
        return RISK_LEVEL.LOW
    if probability < HIGH_RISK_THRESHOLD:
        return RISK_LEVEL.MEDIUM
    return RISK_LEVEL.HIGH


@router.post("/predict")
async def predict_fraud(request: Request,transaction:Transaction):
    model = request.app.state.model
    redis_connection = request.app.state.redis_connection
    db_connection = request.app.state.db_connection

    # transaction to dataframe
    df = pd.DataFrame([transaction.model_dump()])
    df["date"] = pd.to_datetime(df["date"])

    realtime_features = await get_realtime_features(transaction, redis_connection)

    # populate transaction with card/user historic data
    for key, value in realtime_features.items():
        df[key] = value

    prediction = float(model.predict_proba(df)[:, 1][0]) # make a predictiom
    trx_predicted = TransactionPredicted(**transaction.model_dump(),fraud_probability=prediction,risk_level=get_risk_level(prediction),**realtime_features,)

    # save trnsaction to the db
    await save_transaction(trx_predicted, db_connection)
    update_status = await update_redis_cache(transaction, redis_connection)

    return {'result':prediction, 'cache_updated':update_status}


