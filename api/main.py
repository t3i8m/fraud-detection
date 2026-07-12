from contextlib import asynccontextmanager
from fastapi import FastAPI
import mlflow.pyfunc
import uvicorn
from api.routers import model_router

mlflow.set_tracking_uri("http://mlflow:5000")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = mlflow.pyfunc.load_model("models:/fraud-online-xgb@champion")
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(router=model_router.router, prefix="/api/v1/model", tags=["Model inference"])


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)