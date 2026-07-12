from fastapi import APIRouter, Request
import pandas as pd
from api.schemas.transaction_schema import Transaction


router = APIRouter()

@router.post("/predict")
async def predict_fraud(request: Request,transaction:Transaction):
    model = request.app.state.model
    df = pd.read_csv([transaction.model_dump()])

    prediction = model.predict_proba(transaction)
    return {'result':prediction}


