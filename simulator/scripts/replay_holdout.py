import asyncio
import logging
import os
import asyncpg
import httpx
import redis.asyncio as aioredis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INTERVAL_SECONDS = 15
API_URL = "http://api:8000/api/v1/model/predict"
CURSOR_KEY = "simulator:last_id"

redis_connection = aioredis.Redis(host=os.environ["REDIS_HOST"], port=os.environ["REDIS_PORT"], password=os.environ["REDIS_PASSWORD"], decode_responses=True)


async def get_postgre_pool():
    return await asyncpg.create_pool(host=os.environ["POSTGRES_HOST"],port=os.environ["POSTGRES_PORT"],user=os.environ["POSTGRES_USER"],password=os.environ["POSTGRES_PASSWORD"],database=os.environ["POSTGRES_DB"])


async def get_next_transaction(postgre_pool, last_id):
    query = """
    SELECT id, date, client_id, card_id, amount, use_chip, merchant_id, merchant_city, merchant_state, zip, mcc, errors
    FROM holdout_transactions
    WHERE id > $1 AND use_chip = 'Online Transaction'
    ORDER BY id ASC
    LIMIT 1
    """
    return await postgre_pool.fetchrow(query, last_id)


def build_payload(record):
    return {
        "id": record["id"],
        "date": record["date"].isoformat(),
        "client_id": record["client_id"],
        "card_id": record["card_id"],
        "amount": float(record["amount"]),
        "use_chip": record["use_chip"],
        "merchant_id": record["merchant_id"],
        "merchant_city": record["merchant_city"],
        "merchant_state": record["merchant_state"],
        "zip": record["zip"],
        "mcc": record["mcc"],
        "errors": record["errors"],
    }


async def reset_simulation(postgre_pool):
    async with postgre_pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("DELETE FROM predictions WHERE id IN (SELECT id FROM holdout_transactions)")
            await conn.execute("DELETE FROM transactions WHERE id IN (SELECT id FROM holdout_transactions)")
    await redis_connection.delete(CURSOR_KEY)


async def run_forever():
    postgre_pool = await get_postgre_pool()

    async with httpx.AsyncClient() as client:
        while True:
            last_id_raw = await redis_connection.get(CURSOR_KEY)
            last_id = int(last_id_raw) if last_id_raw else 0

            record = await get_next_transaction(postgre_pool, last_id)
            if record is None:
                logger.info("Reached end of holdout data, resetting simulation")
                await reset_simulation(postgre_pool)
                await asyncio.sleep(INTERVAL_SECONDS)
                continue

            payload = build_payload(record)
            try:
                response = await client.post(API_URL, json=payload, timeout=10)
                logger.info(f"Sent transaction {record['id']}, status={response.status_code}, response={response.text}")
            except Exception as ex:
                logger.exception(f"Failed to send transaction {record['id']}, reason: {ex}")

            await redis_connection.set(CURSOR_KEY, record["id"])
            await asyncio.sleep(INTERVAL_SECONDS)


if __name__=="__main__":
    asyncio.run(run_forever())
