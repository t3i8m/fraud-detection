# fraud-detection

Real-time credit card fraud detection: an XGBoost model trained offline on historical
transactions, served behind a FastAPI endpoint that scores live traffic using features
reconstructed from Redis in milliseconds, with drift monitoring and dashboards on top.

Built on the [Kaggle credit card transactions dataset](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets) - 13.3M+ transactions, ~2,000 cardholders, 6,000+ cards.

## How it works

**Offline** - Kaggle CSVs feed feature engineering (`src/data_loader.py`, `src/features.py`)
and an sklearn `Pipeline` (`src/pipeline.py`), trained and cross-validated with
`TimeSeriesSplit` (`src/train.py`), and logged to MLflow: PR curve, feature importance,
a SHAP summary plot. The best run gets promoted to the `fraud-online-xgb@champion` alias
in the MLflow Model Registry.

**Online** - a transaction hits `POST /api/v1/model/predict`. The API loads the champion
model at startup (falling back to a bundled `models/final_model.pkl` if the registry is
unreachable), pulls per-card and per-client state from Redis (1-hour transaction window,
seen merchants/MCCs, running amount stats), scores the transaction, and returns a risk
tier (`LOW` / `MEDIUM` / `HIGH`). The Postgres write and the Redis cache update happen in
a background task, after the response is already on its way back to the client.

**Around it** - a `simulator` replays held-out 2016 transactions through a `frontend` so
you can watch predictions happen; `sync_worker` periodically rebuilds card/client profiles
in Postgres and pushes them to Redis with a 14-day lag (so it never sees the future);
`drift_worker` runs an hourly Evidently `DataDriftPreset` check on recent predictions;
Prometheus scrapes the API and Grafana dashboards sit on top of Prometheus and Postgres.

## Repo layout

```
src/                  feature engineering, sklearn pipeline, training + MLflow logging
  data_loader.py       raw CSVs -> engineered features, train/test split by date
  features.py           FeatureTransformer (target-encoded risk scores, calendar features),
                         RareCategoryGrouper (collapses long-tail categories to "Other")
  pipeline.py            builds the sklearn Pipeline for xgboost/lgbm/catboost/rf/lr
  train.py                training loops (CV + full-fit), MLflow logging, SHAP/PR/importance plots
scripts/
  main.py               entry point: load data, train (CV or full), log to MLflow
  tune.py                 Optuna hyperparameter search
  load_to_postgres.py      one-off: seed Postgres from the raw Kaggle CSVs
api/                   FastAPI service
  main.py                 app setup, model loading, Prometheus middleware
  routers/model_router.py  POST /api/v1/model/predict
  db/repository.py         Postgres writes, Redis feature reads/writes
  schemas/                 Transaction / TransactionPredicted / risk & review enums
sync_worker/            batch job: Postgres -> Redis profile sync (14-day lag)
drift_worker/           hourly Evidently drift check -> model_drift_reports
simulator/              replays holdout transactions against the live stack
frontend/               ingest endpoint + WebSocket relay + live dashboard
monitoring/             Prometheus scrape config + Grafana provisioning
sql/schema.sql          Postgres schema
notebooks/              EDA and error analysis
tests/                  pytest suite (api + repository layer)
```

## Running it locally

```bash
cp .env.example .env        # fill in Postgres/Redis credentials
docker compose up -d        # api, mlflow, postgres, redis, workers, prometheus, grafana
```

| Service    | URL                     | Notes                                    |
|------------|--------------------------|-------------------------------------------|
| API        | http://localhost:8050     | `/api/v1/model/predict`, `/metrics`        |
| Frontend   | http://localhost:8060     | live transaction feed, `/ingest` endpoint  |
| MLflow     | http://localhost:5050     | experiments, runs, model registry          |
| Prometheus | http://localhost:9090     | scrapes `api:8000/metrics` every 15s       |
| Grafana    | http://localhost:3000     | login `admin` / `$GRAFANA_ADMIN_PASSWORD`  |

`docker-compose.prod.yml` mirrors the same services against pre-built GHCR images instead
of local builds, pinned by `APP_VERSION` in `.env`.

Seed Postgres from the raw Kaggle CSVs (expects `data/raw/*.csv`; data isn't committed,
see `.gitignore`):

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python scripts/load_to_postgres.py
```

### Environment variables

| Variable                | Used by                          | Notes                                  |
|--------------------------|-----------------------------------|------------------------------------------|
| `POSTGRES_HOST/PORT/USER/PASSWORD/DB` | api, sync_worker, drift_worker, simulator, grafana | |
| `REDIS_HOST/PORT/PASSWORD`            | api, sync_worker, simulator                        | |
| `APP_VERSION`             | docker-compose.prod.yml            | image tag pulled from GHCR; bumped by the release workflow |
| `GRAFANA_ADMIN_PASSWORD`  | grafana                            | defaults to `admin` if unset            |

## Training a model

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
docker compose up -d mlflow          # or: mlflow server --backend-store-uri sqlite:///mlflow.db
python scripts/main.py               # trains, cross-validates, logs to MLflow
```

`scripts/main.py` runs `train_timeseries_cv` (5-fold `TimeSeriesSplit`, out-of-fold
predictions) or `train_full` (train on the full window, evaluate on the 2016/2019
holdout) depending on the `final_testing` flag near the top of the file. Both paths log
to MLflow: `test`/`cv_*` AUC, PR-AUC, recall, precision, F1, a PR curve, a top-20 feature
importance bar chart, and a SHAP summary plot (`TreeExplainer`/`LinearExplainer` over a
2,000-row sample, so it stays fast even on multi-million-row training sets).

`build_pipeline()` (`src/pipeline.py`) supports five model types via a single switch:
`xgboost`, `lgbm`, `catboost`, `rf` (RandomForest), `lr` (LogisticRegression) - each with
its own sane defaults, all sharing the same preprocessing (median-impute + scale numeric
columns, one-hot encode categoricals, pass everything else through).

`scripts/tune.py` runs an Optuna search (default 50 trials) over the same CV loop,
optimizing PR-AUC, with `scale_pos_weight` derived from the actual class imbalance.

To put a trained model in front of live traffic, promote its MLflow run to the
`fraud-online-xgb@champion` alias in the Model Registry - that's what `api/main.py` loads
on startup.

### Two models, two feature sets

The pipeline branches on `online` vs. `offline` (physical, in-person) transactions -
different fraud patterns, different available signals, different config blocks
(`*_ONLINE` / `*_OFFLINE` in `config.py`). Only the online model is served in production
today; the offline path exists for the notebooks in `notebooks/`.

**Online model features** (`NUMERIC_COLS_ONLINE` + passthrough, no one-hot columns):

| Feature | What it is |
|---|---|
| `amount` | transaction amount |
| `hour` / `day_of_week` / `month` | calendar features from `date` |
| `online_history_ratio` | share of this card's past transactions that were online |
| `time_since_last_trx` | seconds since this card's previous transaction (`-1` if none) |
| `trx_count_1h` / `trx_amount_1h` | count / sum of this card's transactions in the trailing hour |
| `is_new_mcc` | this merchant category is new for the client, as of 14 days ago |
| `user_amount_z_score` | how far this amount is from the client's historical mean, in std-devs |
| `has_insufficient_balance` | this transaction's `errors` field mentions insufficient balance |
| `is_refund` | `amount < 0` |
| `merchant_risk` | target-encoded (`CatBoostEncoder`) fraud rate for this merchant |

Raw identifiers (`id`, `card_id`, `client_id`, `merchant_id`, `merchant_city/state`, `zip`,
`mcc`, `date`, `use_chip`, `errors`) and a handful of engineered columns
(`mcc_risk`, `merchant_state_risk`, `card_swipe_ratio`, `is_new_merchant`, `is_midday`,
the raw error-count columns, `prev_fraud_count[_30days]`) are computed but dropped before
the model sees them - either because they leak identity/label information directly, or
because they didn't earn their keep during feature selection. See `COLS_TO_DROP_ONLINE`
in `config.py` for the exact list.

## Serving predictions

```
POST /api/v1/model/predict
```

Request body (`Transaction`, see `api/schemas/transaction_schema.py`):

```json
{
  "id": 10649266,
  "date": "2016-03-14T09:41:00",
  "client_id": 1556,
  "card_id": 4327,
  "amount": 143.50,
  "use_chip": "Online Transaction",
  "merchant_id": 92103,
  "merchant_city": "ONLINE",
  "merchant_state": null,
  "zip": null,
  "mcc": 5732,
  "errors": null
}
```

Response:

```json
{
  "result": 0.0421,
  "risk_level": "low",
  "binary_prediction": "approved",
  "time_since_last_trx": 5312.0,
  "online_history_ratio": 0.62,
  "is_new_merchant": 0,
  "is_new_mcc": 0,
  "user_amount_z_score": 0.31,
  "trx_count_1h": 1,
  "trx_amount_1h": 89.0,
  "...": "the rest of get_realtime_features()"
}
```

Risk tiers (`api/config.py`) - deliberately three buckets instead of one cutoff, so a
review workflow can route on tier rather than eyeballing a raw probability:

| Tier | Probability | Suggested action |
|---|---|---|
| `low`    | `< 0.205`        | approve automatically |
| `medium` | `0.205 - 0.830`  | flag for review |
| `high`   | `≥ 0.830`        | block / escalate |

`DECISION_THRESHOLD = 0.5` also drives a plain `approved` / `reject` binary field for
anything downstream that wants a single boolean instead of a tier.

The response is sent immediately after `predict_proba`; writing the transaction and
prediction to Postgres and updating the Redis cache both happen in a FastAPI
`BackgroundTask`, after the client already has its answer.

## Data stores

**Redis** (per-card/per-client state, read *and* written on every prediction):

| Key | Holds |
|---|---|
| `card_stats:{card_id}` | last transaction timestamp, total/online transaction counts |
| `trx_window:{card_id}` | sorted set of this card's transactions in the trailing hour |
| `seen_merchant:{client_id}` / `seen_mcc:{client_id}` | sets of merchants/MCCs this client has used |
| `amount_stats:{client_id}` | running count/sum/sum-of-squares for the client's amount z-score |

Keys expire after 90 days. `sync_worker` rebuilds all of the above from Postgres on a
schedule, filtered to `target = 'No'` transactions at least 14 days old - so the cache a
live prediction reads from can never have seen the label of a transaction it's about to
score, or a transaction that happened after it.

**Postgres** (`sql/schema.sql`): `users`, `cards`, `transactions`, `predictions`
(one row per scored transaction, including every feature the model saw),
`model_drift_reports` (one row per `drift_worker` run), and `holdout_transactions`
(the 2016 data the `simulator` replays).

## Monitoring

- Every API request goes through a Prometheus middleware (`http_requests_total`,
  `http_request_duration_seconds`, labeled by method/route/status).
- `drift_worker` runs hourly, compares a recent window of `predictions` against a
  reference window with Evidently's `DataDriftPreset`, and writes dataset-drift /
  drift-share back to `model_drift_reports`.
- Grafana (provisioned in `monitoring/grafana/`) reads both Prometheus and Postgres
  directly, so request volume/latency and drift/risk-tier distribution live on the same
  dashboards.

## Tests

```bash
pytest tests/ -v
```

Covers the risk-tier boundaries and the Postgres repository layer (`tests/`). Runs on
every push/PR via `.github/workflows/tests.yml`. Tagging a commit `v*.*.*` runs the same
suite, then builds and pushes all five images (`api`, `frontend`, `simulator`,
`sync_worker`, `drift_worker`) to GHCR, then SSHes into the prod host, bumps
`APP_VERSION`, and runs `docker compose up -d` - with a `/metrics` health-check and an
automatic rollback to the previous version if it fails
(`.github/workflows/docker-publish.yml`).

## Design notes

A few decisions that aren't obvious from reading any single file:

- **Cold start gets explicit defaults**, not a crash: a card/client with no history yet
  reads `time_since_last_trx = -1`, ratios at `0.0`, rather than failing the lookup.
- **MLflow being down doesn't take down the API.** `load_model()` falls back to a
  `cloudpickle`d snapshot baked into the Docker image (`models/final_model.pkl`) - stale
  the moment a newer model is promoted, but a stale model that answers beats an endpoint
  that won't boot.
- **`sync_worker`'s 14-day lag exists to prevent label leakage**: a client's fraud-derived
  profile in Redis should never reflect a transaction that happened in the last two
  weeks, so a fraudulent transaction can't immediately change how the *next* transaction
  from the same client gets scored.
- **SHAP runs on a 2,000-row sample**, not the full training set - `TreeExplainer` over
  millions of rows would make every training run minutes slower for no material change
  in which features rank where.

## Notebooks

`notebooks/` holds the exploratory work behind the current feature set and model choice:
dataset/EDA passes (`transactions_eda`, `users_eda`, `cards_eda`, `physical_fraud_eda`),
Optuna trial analysis (`optuna_analysis`), and a sequence of error-analysis passes
(`errors_analysis_baseline` -> `errors_analysis_removed_online` ->
`error_analysis_online_model[_no_merchants]` -> `error_analysis_offline_model`) tracking
how removing/adding specific features changed the false-positive/false-negative mix.
`notebooks/eda_notes.md` has the running summary.
