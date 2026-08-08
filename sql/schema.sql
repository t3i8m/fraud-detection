CREATE TABLE users (
    id BIGINT PRIMARY KEY,
    current_age INTEGER,
    retirement_age INTEGER,
    birth_year INTEGER,
    birth_month INTEGER,
    gender TEXT,
    address TEXT,
    latitude NUMERIC,
    longitude NUMERIC,
    per_capita_income NUMERIC(12,2),
    yearly_income NUMERIC(12,2),
    total_debt NUMERIC(12,2),
    credit_score INTEGER,
    num_credit_cards INTEGER
);

CREATE TABLE cards (
    id BIGINT PRIMARY KEY,
    client_id BIGINT NOT NULL REFERENCES users(id),
    card_brand TEXT,
    card_type TEXT,
    card_number TEXT,
    expires TEXT,
    cvv TEXT,
    has_chip TEXT,
    num_cards_issued INTEGER,
    credit_limit NUMERIC(12,2),
    acct_open_date TEXT,
    year_pin_last_changed INTEGER,
    card_on_dark_web TEXT
);

CREATE TYPE review_status AS ENUM ('approved', 'pending', 'reject');

CREATE TABLE transactions (
    id BIGINT PRIMARY KEY,
    date TIMESTAMP NOT NULL,
    client_id BIGINT NOT NULL REFERENCES users(id),
    card_id BIGINT NOT NULL REFERENCES cards(id),
    amount NUMERIC(12,2) NOT NULL,
    use_chip TEXT NOT NULL,
    merchant_id BIGINT NOT NULL,
    merchant_city TEXT,
    merchant_state TEXT,
    zip TEXT,
    mcc INTEGER NOT NULL,
    errors TEXT,
    target TEXT,
    status review_status NOT NULL DEFAULT 'pending'
);

CREATE TABLE predictions (
    id BIGINT PRIMARY KEY REFERENCES transactions(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    fraud_probability NUMERIC(12,2),
    risk_level TEXT,
    time_since_last_trx NUMERIC,
    online_history_ratio NUMERIC,
    is_new_merchant SMALLINT,
    is_new_mcc SMALLINT,
    user_amount_z_score NUMERIC,
    trx_count_1h INTEGER,
    trx_amount_1h NUMERIC,
    has_bad_pin SMALLINT,
    has_insufficient_balance SMALLINT,
    has_technical_glitch SMALLINT,
    bad_pin_count_1h INTEGER,
    bad_cvv_count_1h INTEGER,
    insufficient_balance_count_1h INTEGER,
    tech_glitch_count_1h INTEGER,
    binary_prediction TEXT
);

CREATE TABLE model_drift_reports (
    id SERIAL PRIMARY KEY,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    reference_window_start TIMESTAMPTZ,
    reference_window_end TIMESTAMPTZ,
    current_window_start TIMESTAMPTZ,
    current_window_end TIMESTAMPTZ,
    dataset_drift BOOLEAN,
    drift_share NUMERIC,
    n_features_analyzed INTEGER,
    n_features_drifted INTEGER,
    details JSONB
);

CREATE INDEX idx_predictions_created_at ON predictions (created_at);

CREATE TABLE holdout_transactions (
    id BIGINT PRIMARY KEY,
    date TIMESTAMP NOT NULL,
    client_id BIGINT NOT NULL,
    card_id BIGINT NOT NULL,
    amount NUMERIC(12,2) NOT NULL,
    use_chip TEXT NOT NULL,
    merchant_id BIGINT NOT NULL,
    merchant_city TEXT,
    merchant_state TEXT,
    zip TEXT,
    mcc INTEGER NOT NULL,
    errors TEXT,
    target TEXT
);

CREATE INDEX idx_transactions_card_date ON transactions (card_id, date);
CREATE INDEX idx_transactions_client_date ON transactions (client_id, date);
