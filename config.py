THRESHOLD_ONLINE = 0.5
THRESHOLD_PHYSICAL = 0.5
COLS_TO_DROP_ONLINE = ["id",'card_swipe_ratio',"mcc_risk","merchant_risk","card_type","date","merchant_state_risk","location_type", "prev_fraud_count_30days", "mcc", "client_id", "errors", "customer_id", "card_id", "merchant_id", "merchant_city", "use_chip","is_online",  "merchant_state", "zip", "target", "has_bad_cvv"]
CATEGORIC_COLS_ONLINE = [] # 'mcc_category', "errors"
NUMERIC_COLS_ONLINE = ['amount', 'hour', 'day_of_week', 'month',  "online_history_ratio", "time_since_last_trx", "trx_count_1h", "trx_amount_1h"]

COLS_TO_DROP_OFFLINE = ["day_of_week","is_new_merchant","location_type","distance_z_score","velocity_km_h","mcc","distance_to_home_km","client_id_y","client_id_x",'id','current_age','retirement_age','birth_year',"birth_month","gender","address","latitude","longitude","online_history_ratio","per_capita_income","yearly_income","total_debt","credit_score","num_credit_cards","card_brand","card_number","expires","cvv","num_cards_issued","credit_limit","acct_open_date","year_pin_last_changed","card_on_dark_web","id", "date", "client_id","card_id","longitude", "latitude", "errors", "customer_id", "card_id", "merchant_id", "merchant_city","is_online",  "merchant_state", "zip", "target"]
NUMERIC_COLS_OFFLINE = ["bad_pin_count_1h", 'amount',"tech_glitch_count_1h","bad_cvv_count_1h","insufficient_balance_count_1h","has_technical_glitch","has_insufficient_balance", "has_bad_pin", "has_bad_cvv", 'hour', 'month',  "time_since_last_trx", "trx_count_1h", "trx_amount_1h","mcc_risk"]
CATEGORIC_COLS_OFFLINE= [ 'use_chip',  "has_chip"]
