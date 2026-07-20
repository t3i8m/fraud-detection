
import mlflow
import pandas as pd
from config import THRESHOLD_PHYSICAL, THRESHOLD_ONLINE
from src.data_loader import load_data, save_data
from src.train import train_timeseries_cv, train_full

mlflow.set_tracking_uri("http://localhost:5050")


def main():
    model = "xgboost"
    online_trx = True
    final_testing = False
    exp_type = "ONLINE" if online_trx else "PHYSICAL"
    thr = THRESHOLD_ONLINE if online_trx else THRESHOLD_PHYSICAL
    experiment_name = f"{exp_type}_channelisolated_lag14d_bestparams_colsamplereg_noismcc"

    mode_params = {'max_depth': 6, 'learning_rate': 0.030531371755859415, 'n_estimators': 165, 'subsample': 0.5007308604665407, 'colsample_bytree': 0.4, 'colsample_bylevel': 0.5, 'colsample_bynode': 0.5, 'reg_lambda': 75.13881509487554, 'reg_alpha': 0.7558212888334405, 'min_child_weight': 41, 'gamma': 2.5331823743196136, 'scale_pos_weight': 81.98084102260175}
    feature_weight_map = None

    transactions_df = pd.read_csv("data/raw/transactions_data.csv", parse_dates=["date"])
    fraud_labels = pd.read_json("data/raw/train_fraud_labels.json")
    transactions_df = transactions_df[transactions_df.id.isin(fraud_labels.index)]
    transactions_df = transactions_df.merge(fraud_labels.reset_index().rename(columns={"index": "id", "target": "target"}), on="id")

    train_df, test_df = load_data(transactions_df, online_trx, users_path="data/raw/users_data.csv" , cards_path = "data/raw/cards_data.csv")

    train_df = train_df.sort_values("date").reset_index(drop=True)
    test_df = test_df.sort_values("date").reset_index(drop=True)

    print("Train data was loaded", train_df.shape)
    print(train_df["target"].value_counts(normalize=True))

    print("Test data was loaded", test_df.shape)
    print(test_df["target"].value_counts(normalize=True))

    if final_testing:
        df = test_df.copy()
        y_true, y_hat = train_full(train_df, test_df, model, exp_type, experiment_name, online=online_trx, threshold=thr, model_params=mode_params, feature_weight_map=feature_weight_map)
    else:
        df = train_df.copy()
        y_true, y_hat = train_timeseries_cv(train_df, model, exp_type, experiment_name, online=online_trx, threshold=thr, model_params=mode_params, feature_weight_map=feature_weight_map)

    df["predicted_prob"] = y_hat

    save_data(df, f"data/predicted/predicted_{model}_{experiment_name}.csv")
    print("Done")
    return
    

if __name__=="__main__":
    main()