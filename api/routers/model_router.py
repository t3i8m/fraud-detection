from fastapi import APIRouter, Request
import pandas as pd
from api.db.repository import get_realtime_features, update_redis_cache
from api.schemas.transaction_schema import Transaction


router = APIRouter()

@router.post("/predict")
async def predict_fraud(request: Request,transaction:Transaction):
    model = request.app.state.model
    redis_connection = request.app.state.redis_connection

    # transaction to dataframe
    df = pd.DataFrame([transaction.model_dump()])
    df["date"] = pd.to_datetime(df["date"])

    realtime_features = await get_realtime_features(transaction, redis_connection)

    # populate transaction with card/user historic data
    for key, value in realtime_features.items():
        df[key] = value

    prediction = float(model.predict_proba(df)[:, 1][0])

    update_status = await update_redis_cache(transaction, redis_connection)
    if (update_status):
        return {'result':prediction, 'cache_updated':update_status}
    else:
        return {'result':-1}


