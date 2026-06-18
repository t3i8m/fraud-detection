import mlflow
import mlflow.sklearn
import numpy as np
from src.pipeline import build_pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score
from tqdm import tqdm


def train_timeseries_cv(df, model, threshold=0.3):

    tscv = TimeSeriesSplit(n_splits=5)
    df = df.sort_values("date").reset_index(drop=True)
    X = df.drop(columns=["target"])
    y = (df["target"]=="Yes").astype(int)

    pipeline = build_pipeline(model_type=model, cols_to_drop=["id", "date", "client_id", "customer_id", "card_id", "merchant_id", "merchant_city", "use_chip", "merchant_state", "zip", "mcc", "target", "errors"])

    mlflow.set_experiment(f"fraud_detection_ts_cv")
    with mlflow.start_run(run_name=f"{model}_cv_threshold_{threshold}"):

        oof_preds = np.full(len(X), np.nan)
        fold_auc_scores = []
        fold_pr_auc_scores = []
        fold_recall_scores = []
        fold_precision_scores = []
        fold_f1_scores = []

        for index, (train_id, val_id) in enumerate(tqdm(tscv.split(X, y), desc="Folds")):
            # divide by folds by using array of ids
            train_X, train_y = X.iloc[train_id], y[train_id]
            test_X, test_y = X.iloc[val_id], y[val_id]

            pipeline.fit(train_X, train_y)

            # take probs for positive class
            preds = pipeline.predict_proba(test_X)[:, 1]
            binary_preds = (preds >= threshold).astype(int)

            oof_preds[val_id] = preds

            auc = roc_auc_score(test_y, preds)
            pr_auc = average_precision_score(test_y, preds)
            recall = recall_score(test_y, binary_preds, zero_division=0)
            precision = precision_score(test_y, binary_preds, zero_division=0)
            f1 = f1_score(test_y, binary_preds, zero_division=0)

            fold_auc_scores.append(auc)
            fold_pr_auc_scores.append(pr_auc)
            fold_recall_scores.append(recall)
            fold_precision_scores.append(precision)
            fold_f1_scores.append(f1)

            mlflow.log_metric(f"fold_{index}_auc", auc)
            mlflow.log_metric(f"fold_{index}_pr_auc", pr_auc)
            mlflow.log_metric(f"fold_{index}_recall", recall)
            mlflow.log_metric(f"fold_{index}_precision", precision)
            mlflow.log_metric(f"fold_{index}_f1", f1)

        # since we were using TimeSeriesSplit we do not have preds for the first fold
        valid_idx = ~np.isnan(oof_preds)
        y_true_valid = y[valid_idx]
        oof_preds_valid = oof_preds[valid_idx]
        binary_oof_preds_valid = (oof_preds_valid >= threshold).astype(int)


        global_auc = roc_auc_score(y_true_valid, oof_preds_valid)
        global_pr_auc = average_precision_score(y_true_valid, oof_preds_valid)
        global_recall = recall_score(y_true_valid, binary_oof_preds_valid, zero_division=0)
        global_precision = precision_score(y_true_valid, binary_oof_preds_valid, zero_division=0)
        global_f1 = f1_score(y_true_valid, binary_oof_preds_valid, zero_division=0)
        
        mlflow.log_metric("cv_auc_mean", np.mean(fold_auc_scores))
        mlflow.log_metric("cv_pr_auc_mean", np.mean(fold_pr_auc_scores))
        mlflow.log_metric("cv_recall_mean", np.mean(fold_recall_scores))
        mlflow.log_metric("cv_precision_mean", np.mean(fold_precision_scores))
        mlflow.log_metric("cv_f1_mean", np.mean(fold_f1_scores))

        mlflow.log_metric("global_oof_auc", global_auc)
        mlflow.log_metric("global_oof_pr_auc", global_pr_auc)
        mlflow.log_metric("global_oof_recall", global_recall)
        mlflow.log_metric("global_oof_precision", global_precision)
        mlflow.log_metric("global_oof_f1", global_f1)
        mlflow.log_param("decision_threshold", threshold)

    
    return y_true_valid, oof_preds_valid









