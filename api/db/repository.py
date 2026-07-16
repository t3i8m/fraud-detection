import logging
from asyncpg import Pool
from api.schemas.transaction_schema import TransactionPredicted

logger = logging.getLogger(__name__)


async def save_transaction(trx:TransactionPredicted, db_connection:Pool):
    try:
        insert_query = """
        INSERT INTO predictions(id, fraud_probability, risk_level) VALUES ($1, $2, $3)
        """

        await db_connection.execute(insert_query, trx.id, trx.fraud_probability, trx.risk_level)
        return True
    except Exception as ex:
        logger.exception(f"Failed to write in postgre, reason: {ex}")
        return False


