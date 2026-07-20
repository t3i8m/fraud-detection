import logging
from datetime import datetime
from asyncpg import Pool
from redis.asyncio import Redis
from api.schemas.risk_enum import REVIEW_STATUS
from api.schemas.transaction_schema import Transaction, TransactionPredicted

logger = logging.getLogger(__name__)
CACHE_EXPIRATION = 60 * 60 * 24 * 90
WINDOW_1H_SECONDS = 60 * 60


async def insert_transaction(trx:Transaction, db_connection:Pool):
    try:
        insert_query = """
        INSERT INTO transactions(id, date, client_id, card_id, amount, use_chip, merchant_id, merchant_city, merchant_state, zip, mcc, errors)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        """

        await db_connection.execute(
            insert_query,
            trx.id, trx.date, trx.client_id, trx.card_id, trx.amount, trx.use_chip,
            trx.merchant_id, trx.merchant_city, trx.merchant_state, trx.zip, trx.mcc, trx.errors,
        )
        return True
    except Exception as ex:
        logger.exception(f"Failed to insert transaction, reason: {ex}")
        return False


async def save_transaction(trx:TransactionPredicted, db_connection:Pool):
    try:
        insert_query = """
        INSERT INTO predictions(
            id, fraud_probability, risk_level,
            time_since_last_trx, online_history_ratio, is_new_merchant, is_new_mcc, user_amount_z_score,
            trx_count_1h, trx_amount_1h, has_bad_pin, has_insufficient_balance, has_technical_glitch,
            bad_pin_count_1h, bad_cvv_count_1h, insufficient_balance_count_1h, tech_glitch_count_1h, binary_prediction
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
        """

        await db_connection.execute(
            insert_query,
            trx.id, trx.fraud_probability, trx.risk_level.value,
            trx.time_since_last_trx, trx.online_history_ratio, trx.is_new_merchant, trx.is_new_mcc, trx.user_amount_z_score,
            trx.trx_count_1h, trx.trx_amount_1h, trx.has_bad_pin, trx.has_insufficient_balance, trx.has_technical_glitch,
            trx.bad_pin_count_1h, trx.bad_cvv_count_1h, trx.insufficient_balance_count_1h, trx.tech_glitch_count_1h, trx.binary_prediction,
        )
        return True
    except Exception as ex:
        logger.exception(f"Failed to write in postgre, reason: {ex}")
        return False


async def update_transaction_status(trx_id:int, status:REVIEW_STATUS, db_connection:Pool):
    try:
        update_query = "UPDATE transactions SET status = $1 WHERE id = $2"
        await db_connection.execute(update_query, status.value, trx_id)
        return True
    except Exception as ex:
        logger.exception(f"Failed to update transaction status, reason: {ex}")
        return False


def get_error_flags(trx:Transaction):
    errors = (trx.errors or "").lower()
    return {"has_bad_pin": int("pin" in errors),
        "has_bad_cvv": int("cvv" in errors),
        "has_insufficient_balance": int("balance" in errors),
        "has_technical_glitch": int("glitch" in errors),
    }


async def get_realtime_features(trx:Transaction, redis_connection:Redis):
    card_stats_key = f"card_stats:{trx.card_id}"
    window_key = f"trx_window:{trx.card_id}"
    seen_merchant_key = f"seen_merchant:{trx.client_id}"
    seen_mcc_key = f"seen_mcc:{trx.client_id}"
    amount_stats_key = f"amount_stats:{trx.client_id}"

    now_ts = trx.date.timestamp()
    window_start = now_ts - WINDOW_1H_SECONDS

    card_stats = await redis_connection.hgetall(card_stats_key)
    amount_stats = await redis_connection.hgetall(amount_stats_key)
    is_new_merchant = 0 if await redis_connection.sismember(seen_merchant_key, trx.merchant_id) else 1
    is_new_mcc = 0 if await redis_connection.sismember(seen_mcc_key, trx.mcc) else 1

    await redis_connection.zremrangebyscore(window_key, "-inf", window_start)
    window_members = await redis_connection.zrangebyscore(window_key, window_start, "+inf")
    window_parts = [member.split(":") for member in window_members]
    window_amounts = [float(parts[1]) for parts in window_parts]

    error_flags = get_error_flags(trx)

    if card_stats:
        last_trx_ts = datetime.fromisoformat(card_stats["last_trx_ts"])
        total_count = int(card_stats["total_count"])
        online_count = int(card_stats["online_count"])
        time_since_last_trx = (trx.date - last_trx_ts).total_seconds()
        online_history_ratio = online_count / total_count if total_count else 0.0
    else:
        time_since_last_trx = -1.0
        online_history_ratio = 0.0

    if amount_stats:
        count = int(amount_stats["count"])
        amount_sum = float(amount_stats["sum"])
        amount_sumsq = float(amount_stats["sumsq"])
        mean = amount_sum / count if count else 0.0
        var = amount_sumsq / count - mean ** 2 if count else 0.0
        std = var ** 0.5 if var > 0 else 1.0
    else:
        mean, std = 0.0, 1.0

    return {"time_since_last_trx": time_since_last_trx,
        "online_history_ratio": online_history_ratio,
        "is_new_merchant": is_new_merchant,
        "is_new_mcc": is_new_mcc,
        "user_amount_z_score": (trx.amount - mean) / std,
        "trx_count_1h": len(window_amounts),
        "trx_amount_1h": sum(window_amounts),
        "has_bad_pin": error_flags["has_bad_pin"],
        "has_insufficient_balance": error_flags["has_insufficient_balance"],
        "has_technical_glitch": error_flags["has_technical_glitch"],
        "bad_pin_count_1h": sum(int(parts[2]) for parts in window_parts),
        "bad_cvv_count_1h": sum(int(parts[3]) for parts in window_parts),
        "insufficient_balance_count_1h": sum(int(parts[4]) for parts in window_parts),
        "tech_glitch_count_1h": sum(int(parts[5]) for parts in window_parts),
    }


async def update_redis_cache(trx:Transaction, redis_connection:Redis):
    try:
        card_stats_key = f"card_stats:{trx.card_id}"
        window_key = f"trx_window:{trx.card_id}"
        is_online = trx.use_chip == "Online Transaction"

        await redis_connection.hset(card_stats_key, "last_trx_ts", trx.date.isoformat())
        await redis_connection.hincrby(card_stats_key, "total_count", 1)
        if is_online:
            await redis_connection.hincrby(card_stats_key, "online_count", 1)
        await redis_connection.expire(card_stats_key, CACHE_EXPIRATION)

        error_flags = get_error_flags(trx)
        window_member = f"{trx.id}:{trx.amount}:{error_flags['has_bad_pin']}:{error_flags['has_bad_cvv']}:{error_flags['has_insufficient_balance']}:{error_flags['has_technical_glitch']}"
        await redis_connection.zadd(window_key, {window_member: trx.date.timestamp()})
        await redis_connection.expire(window_key, WINDOW_1H_SECONDS)
        
        return True
    except Exception as ex:
        logger.exception(f"Failed to update redis cache, reason: {ex}")
        return False
