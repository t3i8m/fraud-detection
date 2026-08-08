import logging
from fastapi import APIRouter, BackgroundTasks, Request
import pandas as pd
from api.config import DECISION_THRESHOLD, HIGH_RISK_THRESHOLD, LOW_RISK_THRESHOLD
from api.db.repository import get_realtime_features, update_redis_cache, save_transaction, insert_transaction
from api.schemas.risk_enum import RISK_LEVEL
from api.schemas.transaction_schema import Transaction, TransactionPredicted

logger = logging.getLogger(__name__)

router = APIRouter()


def get_risk_level(probability):
    if probability < LOW_RISK_THRESHOLD:
        return RISK_LEVEL.LOW
    if probability < HIGH_RISK_THRESHOLD:
        return RISK_LEVEL.MEDIUM
    return RISK_LEVEL.HIGH


async def _persist_prediction(transaction: Transaction, trx_predicted: TransactionPredicted, db_connection, redis_connection):
    # runs after the response is already on its way to the client 
    await insert_transaction(transaction, db_connection)
    await save_transaction(trx_predicted, db_connection)
    await update_redis_cache(transaction, redis_connection)


@router.post("/predict")
async def predict_fraud(request: Request, transaction: Transaction, background_tasks: BackgroundTasks):
    model = request.app.state.model
    redis_connection = request.app.state.redis_connection
    db_connection = request.app.state.db_connection

    logger.info(f"Received a new transaction with id: {transaction.id}")

    # transaction to dataframe
    df = pd.DataFrame([transaction.model_dump()])
    df["date"] = pd.to_datetime(df["date"])

    realtime_features = await get_realtime_features(transaction, redis_connection)

    # populate transaction with card/user historic data
    for key, value in realtime_features.items():
        df[key] = value

    logger.info(f"Classifying... {transaction.id}")

    prediction = float(model.predict_proba(df)[:, 1][0]) # make a prediction
    binary_prediction = "reject" if prediction > DECISION_THRESHOLD else "approved"
    risk_level = get_risk_level(prediction)
    trx_predicted = TransactionPredicted(**transaction.model_dump(),fraud_probability=prediction,risk_level=risk_level,binary_prediction=binary_prediction,**realtime_features,)
    logger.info(f"Transaction {transaction.id} classified as {binary_prediction} (probability={prediction:.4f}, risk_level={risk_level.value})")

    background_tasks.add_task(_persist_prediction, transaction, trx_predicted, db_connection, redis_connection)

    return {
        'result': prediction,
        'risk_level': risk_level.value,
        'binary_prediction': binary_prediction,
        **realtime_features,
    }


