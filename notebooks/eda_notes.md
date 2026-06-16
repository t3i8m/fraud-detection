# EDA Summary - transactions_eda.ipynb

## Dataset / split
- ~13.3M transactions; ~8.9M have fraud labels (rest = test/submission set).
- Train/test split by date (cutoff `2018-01-01`) to simulate production.
- Highly imbalanced target: ~0.14% fraud in both train and test.

## Missing values / cleaning
- `merchant_state` and `zip` missing (~12%) almost entirely correspond to `merchant_city == "ONLINE"` -> missing city means online purchase.
- `errors` missing (~98%) -> impute as `"No Error"`.
- Missing `zip` -> impute with mode zip of the same `merchant_city`.
- Rare categories in categorical columns -> bucket into `"Other"`.

## Engineered feature ideas
- **is_refund**: `amount < 0` (refund). Fraud refunds have ~2x higher median amount than legit refunds.
- **hour / is_midday**: most fraud occurs during daytime (10:00-16:00) -> add `hour` feature or `is_midday` flag.
- **prev_fraud_history**: number of prior frauds on a card (if card had fraud once, repeat probability increases). Must be built carefully to avoid data leakage.
- **is_online**: derived from `use_chip == "Online Transaction"` / `merchant_city == "ONLINE"`; can replace `use_chip`.
- **is_international**: `merchant_state` not a US state -> higher fraud risk for some intl locations.
- **has_bad_cvv**: `errors` contains `"Bad CVV"` -> highest fraud rate among error types.

## Key observations per feature
- **amount**: fraud transactions have higher mean/median amount than legit; both distributions right-skewed; fraud max ~5k vs legit max ~5.7k. Moderate effect size (rank-biserial -0.376).
- **mcc**: moderate effect size (0.237); specific MCC codes (e.g., 4411, 5733) show much higher fraud rates.
- **hour / day_of_week**: weak effect (rank-biserial ~0.12 / -0.10).
- **year**: fraud rate varies by year -> drop year as a feature.
- **use_chip**: online transactions are the smallest share but have the highest fraud rate.
- **merchant_city**: highest Cramer's V (0.217) among categoricals; e.g., Rome ~9% fraud rate.
- **merchant_state**: second highest Cramer's V (0.184); used for `is_international` feature.
- **errors**: low effect size (0.026) overall, but "Bad CVV" stands out with high fraud rate.
- **zip**: very weak signal, mostly noise -> candidate for removal.
- **card_id**: one card (14 uses, 11 fraud, ratio 0.79) is a potential outlier; no median-amount difference between fraud/non-fraud cards; repeat-fraud cards are common (87% of fraud cards have >1 fraud).
- **Multicollinearity**: Spearman correlation among numeric features shows no significant multicollinearity.

## Open items / TODO
- Decide handling of `zip` (likely drop).
- Build `prev_fraud_history` without leakage (use only past transactions per card).
- Finalize `"Other"` category thresholds for `merchant_city` / `merchant_state` / `errors`.
