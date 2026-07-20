from contextlib import asynccontextmanager
import logging
import os
from fastapi import FastAPI
import mlflow.sklearn
import uvicorn
from api.routers import model_router
import asyncpg
import redis.asyncio as aioredis

logging.basicConfig(level=logging.INFO)
mlflow.set_tracking_uri("http://mlflow:5000")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = mlflow.sklearn.load_model("models:/fraud-online-xgb@champion")
    app.state.db_connection = await asyncpg.create_pool(host=os.environ["POSTGRES_HOST"],port=os.environ["POSTGRES_PORT"],user=os.environ["POSTGRES_USER"],password=os.environ["POSTGRES_PASSWORD"],database=os.environ["POSTGRES_DB"],)
    app.state.redis_connection = aioredis.Redis(host=os.environ["REDIS_HOST"], port=os.environ["REDIS_PORT"], password=os.environ["REDIS_PASSWORD"], decode_responses=True)
    yield

    await app.state.db_connection.close()
    await app.state.redis_connection.close()


app = FastAPI(lifespan=lifespan)
app.include_router(router=model_router.router, prefix="/api/v1/model", tags=["Model inference"])


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)