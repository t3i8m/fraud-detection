import asyncio
import logging
import os
import random
import asyncpg
import httpx
import redis.asyncio as aioredis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INTERVAL_SECONDS = 2
FRONTEND_INGEST_URL = "http://frontend:8000/ingest"
FRAUD_RATIO = 0.2
CURSOR_KEY_FRAUD = "simulator:last_id:fraud"
CURSOR_KEY_NONFRAUD = "simulator:last_id:nonfraud"

redis_connection = aioredis.Redis(host=os.environ["REDIS_HOST"], port=os.environ["REDIS_PORT"], password=os.environ["REDIS_PASSWORD"], decode_responses=True)


async def get_postgre_pool():
    return await asyncpg.create_pool(host=os.environ["POSTGRES_HOST"],port=os.environ["POSTGRES_PORT"],user=os.environ["POSTGRES_USER"],password=os.environ["POSTGRES_PASSWORD"],database=os.environ["POSTGRES_DB"])


async def get_next_transaction(postgre_pool, last_id, target):
    query = """
    SELECT id, date, client_id, card_id, amount, use_chip, merchant_id, merchant_city, merchant_state, zip, mcc, errors, target
    FROM holdout_transactions
    WHERE id > $1 AND use_chip = 'Online Transaction' AND target = $2
    ORDER BY id ASC
    LIMIT 1
    """
    return await postgre_pool.fetchrow(query, last_id, target)


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
        "target": record["target"],
    }


async def reset_simulation(postgre_pool):
    async with postgre_pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("DELETE FROM predictions WHERE id IN (SELECT id FROM holdout_transactions)")
            await conn.execute("DELETE FROM transactions WHERE id IN (SELECT id FROM holdout_transactions)")
    await redis_connection.delete(CURSOR_KEY_FRAUD, CURSOR_KEY_NONFRAUD)


async def run_forever():
    postgre_pool = await get_postgre_pool()

    async with httpx.AsyncClient() as client:
        while True:
            last_fraud_id = int(await redis_connection.get(CURSOR_KEY_FRAUD) or 0)
            last_nonfraud_id = int(await redis_connection.get(CURSOR_KEY_NONFRAUD) or 0)

            target = "Yes" if random.random() < FRAUD_RATIO else "No"
            last_id = last_fraud_id if target == "Yes" else last_nonfraud_id
            record = await get_next_transaction(postgre_pool, last_id, target)

            if record is None:
                # this target is exhausted for now, fall back to the other one
                target = "No" if target == "Yes" else "Yes"
                last_id = last_nonfraud_id if target == "No" else last_fraud_id
                record = await get_next_transaction(postgre_pool, last_id, target)

            if record is None:
                logger.info("Reached end of holdout data, resetting simulation")
                await reset_simulation(postgre_pool)
                await asyncio.sleep(INTERVAL_SECONDS)
                continue

            payload = build_payload(record)
            try:
                response = await client.post(FRONTEND_INGEST_URL, json=payload, timeout=10)
                logger.info(f"Sent transaction {record['id']} to frontend, status={response.status_code}, response={response.text}")
            except Exception as ex:
                logger.exception(f"Failed to send transaction {record['id']}, reason: {ex}")

            cursor_key = CURSOR_KEY_FRAUD if record["target"] == "Yes" else CURSOR_KEY_NONFRAUD
            await redis_connection.set(cursor_key, record["id"])
            await asyncio.sleep(INTERVAL_SECONDS)


if __name__=="__main__":
    asyncio.run(run_forever())
