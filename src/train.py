import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
from config import CATEGORIC_COLS_OFFLINE, CATEGORIC_COLS_ONLINE, COLS_TO_DROP_OFFLINE, COLS_TO_DROP_ONLINE, NUMERIC_COLS_OFFLINE, NUMERIC_COLS_ONLINE
from src.pipeline import build_pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score,precision_recall_curve, recall_score, f1_score
from tqdm import tqdm


def train_timeseries_cv(df, model, experiment_type = "", experiment_name="", online=True, threshold=0.5, model_params={}):

    tscv = TimeSeriesSplit(n_splits=5)
    df = df.sort_values("date").reset_index(drop=True)

    X = df.drop(columns=["target"])
    y = (df["target"]=="Yes").astype(int)

    if online:
        pipeline = build_pipeline(model_type=model, numeric_cols=NUMERIC_COLS_ONLINE, categoric_cols=CATEGORIC_COLS_ONLINE, cols_to_drop=COLS_TO_DROP_ONLINE, model_params=model_params)
    else:
        pipeline = build_pipeline(model_type=model,numeric_cols=NUMERIC_COLS_OFFLINE, categoric_cols=CATEGORIC_COLS_OFFLINE, cols_to_drop=COLS_TO_DROP_OFFLINE, model_params=model_params)


    mlflow.set_experiment(f"{experiment_type}_fraud_detection_ts_cv")
    with mlflow.start_run(run_name=f"{model}_cv_{experiment_name}_threshold_{threshold}"):

        oof_preds = np.full(len(X), np.nan)
        fold_auc_scores = []
        fold_pr_auc_scores = []
        fold_recall_scores = []
        fold_precision_scores = []
        fold_f1_scores = []

        fold_train_auc_scores = []
        fold_train_pr_auc_scores = []

        for index, (train_id, val_id) in enumerate(tqdm(tscv.split(X, y), desc="Folds")):
            # divide by folds by using array of ids
            train_X, train_y = X.iloc[train_id], y[train_id]
            test_X, test_y = X.iloc[val_id], y[val_id]

            pipeline.fit(train_X, train_y)

            #  take probs for trining
            train_preds = pipeline.predict_proba(train_X)[:, 1]
            train_auc = roc_auc_score(train_y, train_preds)
            train_pr_auc = average_precision_score(train_y, train_preds)
            fold_train_auc_scores.append(train_auc)
            fold_train_pr_auc_scores.append(train_pr_auc)

            mlflow.log_metric(f"fold_{index}_train_auc", train_auc)
            mlflow.log_metric(f"fold_{index}_train_pr_auc", train_pr_auc)

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
        oof_preds_series = pd.Series(oof_preds, index=X.index, name="predicted_prob")

        y_true_valid = y[valid_idx]
        oof_preds_valid = oof_preds_series[valid_idx]
        binary_oof_preds_valid = (oof_preds_valid >= threshold).astype(int)

        global_auc = roc_auc_score(y_true_valid, oof_preds_valid)
        global_pr_auc = average_precision_score(y_true_valid, oof_preds_valid)
        global_recall = recall_score(y_true_valid, binary_oof_preds_valid, zero_division=0)
        global_precision = precision_score(y_true_valid, binary_oof_preds_valid, zero_division=0)
        global_f1 = f1_score(y_true_valid, binary_oof_preds_valid, zero_division=0)

        mlflow.log_metric("cv_train_auc_mean", np.mean(fold_train_auc_scores))
        mlflow.log_metric("cv_train_pr_auc_mean", np.mean(fold_train_pr_auc_scores))

        mlflow.log_metric("cv_auc_mean", np.mean(fold_auc_scores))
        mlflow.log_metric("cv_pr_auc_mean", np.mean(fold_pr_auc_scores))
        mlflow.log_metric("cv_pr_auc_std", np.std(fold_pr_auc_scores))

        mlflow.log_metric("cv_recall_mean", np.mean(fold_recall_scores))
        mlflow.log_metric("cv_precision_mean", np.mean(fold_precision_scores))
        mlflow.log_metric("cv_f1_mean", np.mean(fold_f1_scores))

        mlflow.log_metric("global_oof_auc", global_auc)
        mlflow.log_metric("global_oof_pr_auc", global_pr_auc)
        mlflow.log_metric("global_oof_recall", global_recall)
        mlflow.log_metric("global_oof_precision", global_precision)
        mlflow.log_metric("global_oof_f1", global_f1)
        mlflow.log_param("decision_threshold", threshold)
        mlflow.sklearn.log_model(pipeline, name="model", serialization_format="cloudpickle")


        log_pr_curve(y_true_valid, oof_preds_valid, global_pr_auc)
        get_feature_importances(pipeline, X, y, model_type=model)

    
    return y_true_valid, oof_preds_valid


def get_feature_importances(pipeline, X, y, model_type):

    pipeline.fit(X, y)
    model_instance = pipeline.named_steps["classifier"]
    
    if hasattr(model_instance, "feature_importances_"):

        feature_names = pipeline[:-1].get_feature_names_out()
        importances = model_instance.feature_importances_
        
        fi_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False).head(20) 
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x="importance", y="feature", data=fi_df, palette="viridis")
        plt.title(f"Top 20 Feature Importance ({model_type})")
        plt.tight_layout()
        
        plot_path = "feature_importance.png"
        plt.savefig(plot_path)
        plt.close()
        
        mlflow.log_artifact(plot_path)
        
        if os.path.exists(plot_path):
            os.remove(plot_path)
    return

def log_pr_curve(y_true, y_prob, pr_auc_score):

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, label=f'OOF PR Curve (AUC = {pr_auc_score:.4f})', color='#2b5c8f', lw=2)

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Global OOF Precision-Recall Curve', fontsize=14, pad=15)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    plot_path = "pr_curve.png"
    plt.savefig(plot_path)
    plt.close()

    mlflow.log_artifact(plot_path)

    if os.path.exists(plot_path):
        os.remove(plot_path)








