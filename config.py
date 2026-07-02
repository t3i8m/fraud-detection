THRESHOLD_ONLINE = 0.99
THRESHOLD_PHYSICAL = 0.5
COLS_TO_DROP = ["id", "date", "client_id", "errors", "customer_id", "card_id", "merchant_id", "merchant_city", "use_chip","is_online",  "merchant_state", "zip", "target", "has_bad_cvv"]
CATEGORIC_COLS = [] # 'mcc_category', "errors"
NUMERIC_COLS = ['amount', 'hour', 'day_of_week', 'month', 'prev_fraud_count', "online_history_ratio", "time_since_last_trx", "trx_count_1h", "trx_amount_1h"]