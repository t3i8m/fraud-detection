import asyncio
import logging
import asyncpg
import os
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)
redis_connection = aioredis.Redis(host=os.environ["REDIS_HOST"], port=os.environ["REDIS_PORT"], password=os.environ["REDIS_PASSWORD"], decode_responses=True)

EXPIRATION = 60 * 60 * 24 * 90 # we set max 90 days to store clients data

async def get_postgre_pool():
    return await asyncpg.create_pool(host=os.environ["POSTGRES_HOST"],port=os.environ["POSTGRES_PORT"],user=os.environ["POSTGRES_USER"],password=os.environ["POSTGRES_PASSWORD"],database=os.environ["POSTGRES_DB"])


async def get_users_data_batch(postgre_pool):
    sql_query = """WITH latest_ts AS (SELECT max(date) AS ts FROM transactions)
    SELECT
        t.client_id,
        array_agg(DISTINCT t.merchant_id) AS seen_merchants,
        array_agg(DISTINCT t.mcc) AS seen_mccs,
        count(t.amount) AS trx_count,
        sum(t.amount) AS amount_sum,
        sum(t.amount * t.amount) AS amount_sumsq
    FROM transactions t, latest_ts
    WHERE t.use_chip = 'Online Transaction'
        AND t.target = 'No'
        AND t.date <= latest_ts.ts - interval '14 days'
    GROUP BY t.client_id
    """
    try:
        records = await postgre_pool.fetch(sql_query)
        return records
    except Exception as ex:
        logger.exception(f"Failed to fetch users data batch, reason: {ex}")
        return []


async def get_cards_activity_batch(postgre_pool):
    sql_query = """WITH latest_ts AS (SELECT max(date) AS ts FROM transactions)
    SELECT
        card_id,
        max(date) AS last_trx_ts,
        count(*) FILTER (WHERE use_chip = 'Online Transaction' AND target = 'No' AND date <= latest_ts.ts - interval '14 days') AS online_count,
        count(*) FILTER (WHERE target = 'No' AND date <= latest_ts.ts - interval '14 days') AS total_count
    FROM transactions, latest_ts
    GROUP BY card_id
    """
    try:
        records = await postgre_pool.fetch(sql_query)
        return records
    except Exception as ex:
        logger.exception(f"Failed to fetch cards activity batch, reason: {ex}")
        return []


async def sync_behaviour():
    postgre_connection = await get_postgre_pool()
    
    try:
        users_records = await get_users_data_batch(postgre_connection)
        cards_records = await get_cards_activity_batch(postgre_connection)

        pipe = redis_connection.pipeline()
        for record in users_records:
            client_id = record["client_id"]
            seen_merchant_key = f"seen_merchant:{client_id}"
            seen_mcc_key = f"seen_mcc:{client_id}"
            amount_stats_key = f"amount_stats:{client_id}"

            pipe.sadd(seen_merchant_key, *record["seen_merchants"])
            pipe.expire(seen_merchant_key, EXPIRATION)

            pipe.sadd(seen_mcc_key, *record["seen_mccs"])
            pipe.expire(seen_mcc_key, EXPIRATION)

            pipe.hset(amount_stats_key, mapping={"count": record["trx_count"],"sum": float(record["amount_sum"]),"sumsq": float(record["amount_sumsq"]),})
            pipe.expire(amount_stats_key, EXPIRATION)

        for record in cards_records:
            card_stats_key = f"card_stats:{record['card_id']}"

            pipe.hset(card_stats_key, mapping={"last_trx_ts": record["last_trx_ts"].isoformat(),"online_count": record["online_count"],"total_count": record["total_count"],})
            pipe.expire(card_stats_key, EXPIRATION)

        await pipe.execute()
    except Exception as ex:
        logger.exception(f"Failed to sync behaviour to redis, reason: {ex}")
    finally:
        await postgre_connection.close()


if __name__=="__main__":
    asyncio.run(sync_behaviour())