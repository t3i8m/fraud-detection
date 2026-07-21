import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import psycopg2
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RUN_INTERVAL_HOURS = float(os.environ.get("DRIFT_RUN_INTERVAL_HOURS", 1))
REFERENCE_WINDOW_HOURS = float(os.environ.get("DRIFT_REFERENCE_WINDOW_HOURS", 2))
CURRENT_WINDOW_HOURS = float(os.environ.get("DRIFT_CURRENT_WINDOW_HOURS", 1))
MIN_ROWS = int(os.environ.get("DRIFT_MIN_ROWS", 30))

FEATURE_COLUMNS = [
    "fraud_probability",
    "time_since_last_trx",
    "online_history_ratio",
    "is_new_merchant",
    "is_new_mcc",
    "user_amount_z_score",
    "trx_count_1h",
    "trx_amount_1h",
]


def get_connection():
    return psycopg2.connect(
        host=os.environ["POSTGRES_HOST"],
        port=os.environ["POSTGRES_PORT"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        dbname=os.environ["POSTGRES_DB"],
    )


def fetch_window(conn, start, end):
    query = f"""
        SELECT {", ".join(FEATURE_COLUMNS)}
        FROM predictions
        WHERE created_at >= %s AND created_at < %s
    """
    return pd.read_sql(query, conn, params=(start, end))


def run_drift_check():
    now = datetime.now(timezone.utc)
    current_start = now - timedelta(hours=CURRENT_WINDOW_HOURS)
    reference_end = current_start
    reference_start = reference_end - timedelta(hours=REFERENCE_WINDOW_HOURS)

    conn = get_connection()
    try:
        reference_df = fetch_window(conn, reference_start, reference_end)
        current_df = fetch_window(conn, current_start, now)

        if len(reference_df) < MIN_ROWS or len(current_df) < MIN_ROWS:
            logger.info(
                f"Not enough data yet for a drift check (reference={len(reference_df)}, "
                f"current={len(current_df)}, need >= {MIN_ROWS} each), skipping this run"
            )
            return

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_df, current_data=current_df)
        result = report.as_dict()
        summary = result["metrics"][0]["result"]

        dataset_drift = summary.get("dataset_drift")
        drift_share = summary.get("share_of_drifted_columns", summary.get("drift_share"))
        n_features = summary.get("number_of_columns", len(FEATURE_COLUMNS))
        n_drifted = summary.get("number_of_drifted_columns")

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO model_drift_reports
                    (reference_window_start, reference_window_end, current_window_start, current_window_end,
                     dataset_drift, drift_share, n_features_analyzed, n_features_drifted, details)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    reference_start, reference_end, current_start, now,
                    dataset_drift, drift_share, n_features, n_drifted,
                    json.dumps(result, default=str),
                ),
            )
        conn.commit()
        logger.info(f"Drift report saved: dataset_drift={dataset_drift}, drift_share={drift_share}")
    except Exception:
        logger.exception("Failed to compute drift report")
    finally:
        conn.close()


async def run_forever():
    while True:
        run_drift_check()
        await asyncio.sleep(RUN_INTERVAL_HOURS * 60 * 60)


if __name__ == "__main__":
    asyncio.run(run_forever())
