import json
import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score
from config import CATEGORIC_COLS_OFFLINE, CATEGORIC_COLS_ONLINE, COLS_TO_DROP_OFFLINE, COLS_TO_DROP_ONLINE, NUMERIC_COLS_OFFLINE, NUMERIC_COLS_ONLINE
from src.data_loader import load_data
from src.pipeline import build_pipeline

online_trx = True
n_trials = 50

train_path = "data/processed/train_online.csv" if online_trx else "data/processed/train_physical.csv"
df = load_data(train_path, online_trx, users_path="data/raw/users_data.csv", cards_path="data/raw/cards_data.csv")
df = df.sort_values("date").reset_index(drop=True)

X = df.drop(columns=["target"])
y = (df["target"] == "Yes").astype(int)

negative_count = (y == 0).sum()
positive_count = (y == 1).sum()
base_scale_pos = round(negative_count / positive_count)

numeric_cols = NUMERIC_COLS_ONLINE if online_trx else NUMERIC_COLS_OFFLINE
categoric_cols = CATEGORIC_COLS_ONLINE if online_trx else CATEGORIC_COLS_OFFLINE
cols_to_drop = COLS_TO_DROP_ONLINE if online_trx else COLS_TO_DROP_OFFLINE

test_size = len(X) // 7
tscv = TimeSeriesSplit(n_splits=5, test_size=test_size)


def cv_pr_auc(model_params):
    pipeline = build_pipeline(model_type="xgboost", numeric_cols=numeric_cols, categoric_cols=categoric_cols, cols_to_drop=cols_to_drop, model_params=model_params)

    oof_preds = np.full(len(X), np.nan)
    for train_id, val_id in tscv.split(X, y):
        train_X, train_y = X.iloc[train_id], y.iloc[train_id]
        val_X = X.iloc[val_id]

        pipeline.fit(train_X, train_y)
        oof_preds[val_id] = pipeline.predict_proba(val_X)[:, 1]

    valid_idx = ~np.isnan(oof_preds)
    return average_precision_score(y[valid_idx], oof_preds[valid_idx])


def objective(trial):
    model_params = {
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 100, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 100, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", base_scale_pos * 0.5, base_scale_pos * 2.0),
    }
    return cv_pr_auc(model_params)


def main():
    exp_type = "online" if online_trx else "physical"
    study = optuna.create_study(direction="maximize", study_name=f"xgboost_{exp_type}", storage="sqlite:///tune_studies.db", load_if_exists=True)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("Best PR-AUC:", study.best_value)
    print("Best params:", study.best_params)

    with open(f"data/predicted/best_params_{exp_type}.json", "w") as f:
        json.dump(study.best_params, f, indent=2)


if __name__ == "__main__":
    main()
