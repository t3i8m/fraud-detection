from contextlib import asynccontextmanager
import logging
import os
import time
import cloudpickle
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import mlflow.sklearn
import uvicorn
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST, REGISTRY
from api.routers import model_router
import asyncpg
import redis.asyncio as aioredis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
mlflow.set_tracking_uri("http://mlflow:5000")

FALLBACK_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "final_model.pkl")


def load_model():
    try:
        model = mlflow.sklearn.load_model("models:/fraud-online-xgb@champion")
        logger.info("Loaded model from MLflow registry (fraud-online-xgb@champion)")
        return model
    except Exception:
        logger.exception("Could not load model from MLflow, falling back to local models/final_model.pkl")
        with open(FALLBACK_MODEL_PATH, "rb") as f:
            return cloudpickle.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    app.state.db_connection = await asyncpg.create_pool(host=os.environ["POSTGRES_HOST"],port=os.environ["POSTGRES_PORT"],user=os.environ["POSTGRES_USER"],password=os.environ["POSTGRES_PASSWORD"],database=os.environ["POSTGRES_DB"],)
    app.state.redis_connection = aioredis.Redis(host=os.environ["REDIS_HOST"], port=os.environ["REDIS_PORT"], password=os.environ["REDIS_PASSWORD"], decode_responses=True)
    yield

    await app.state.db_connection.close()
    await app.state.redis_connection.close()


def _counter(name, documentation, labelnames):
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return Counter(name, documentation, labelnames)


def _histogram(name, documentation, labelnames):
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return Histogram(name, documentation, labelnames)


REQUEST_COUNT = _counter("http_requests_total", "Total HTTP requests", ["method", "handler", "status"])
REQUEST_LATENCY = _histogram("http_request_duration_seconds", "HTTP request latency in seconds", ["method", "handler"])

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],)
app.include_router(router=model_router.router, prefix="/api/v1/model", tags=["Model inference"])


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    handler = request.url.path
    REQUEST_COUNT.labels(request.method, handler, response.status_code).inc()
    REQUEST_LATENCY.labels(request.method, handler).observe(duration)
    return response


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000)