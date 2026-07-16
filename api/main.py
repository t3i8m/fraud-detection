from contextlib import asynccontextmanager
import os
from fastapi import FastAPI
import mlflow.pyfunc
import uvicorn
from api.routers import model_router
import asyncpg

mlflow.set_tracking_uri("http://mlflow:5000")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = mlflow.pyfunc.load_model("models:/fraud-online-xgb@champion")
    app.state.db_connection = await asyncpg.create_pool(host=os.environ["POSTGRES_HOST"],port=os.environ["POSTGRES_PORT"],user=os.environ["POSTGRES_USER"],password=os.environ["POSTGRES_PASSWORD"],database=os.environ["POSTGRES_DB"],)
    yield

    await app.state.db_pool.close()


app = FastAPI(lifespan=lifespan)
app.include_router(router=model_router.router, prefix="/api/v1/model", tags=["Model inference"])


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)