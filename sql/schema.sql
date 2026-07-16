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
    client_id BIGINT NOT NULL,
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

CREATE TABLE transactions (
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

CREATE TABLE predictions (
    id BIGINT PRIMARY KEY REFERENCES transactions(id),
    fraud_probability NUMERIC(12,2),
    risk_level TEXT
);

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
