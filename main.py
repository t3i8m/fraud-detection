
import pandas as pd
from config import THRESHOLD_PHYSICAL, THRESHOLD_ONLINE
from src.data_loader import load_data, save_data
from src.train import train_timeseries_cv, train_full


def main():
    model = "xgboost"
    online_trx = True
    final_testing = True
    exp_type = "ONLINE" if online_trx else "PHYSICAL"
    thr = THRESHOLD_ONLINE if online_trx else THRESHOLD_PHYSICAL
    experiment_name = f"{exp_type}_regularization_scale_pos_2860_added_fraud_count_maxdepth4_estimators150_added_added_behaviour_smoothening2000_regularixzation_nomccrisk_bestparams"

    # best params based on optuna
    mode_params = {'max_depth': 6, 'learning_rate': 0.030531371755859415, 'n_estimators': 165, 'subsample': 0.5007308604665407, 'colsample_bytree': 0.7974378806404453, 'reg_lambda': 75.13881509487554, 'reg_alpha': 0.7558212888334405, 'min_child_weight': 41, 'gamma': 2.5331823743196136, 'scale_pos_weight': 81.98084102260175}

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

    # if model in ["xgboost", "catboost", "lightgbm"]:
    #     negative_count = train_df[train_df["target"]=="No"].shape[0]
    #     positive_count =  train_df[train_df["target"]=="Yes"].shape[0]
    #     scale_pos =  round(negative_count/positive_count)
    #     print("Scale pos:",scale_pos)
    #     mode_params["scale_pos_weight"]=scale_pos

    if final_testing:
        df = test_df.copy()
        y_true, y_hat = train_full(train_df, test_df, model, exp_type, experiment_name,online=online_trx,threshold=thr, model_params=mode_params)
    else:
        df = train_df.copy()
        y_true, y_hat = train_timeseries_cv(train_df, model, exp_type, experiment_name,online=online_trx,threshold=thr, model_params=mode_params)

    df["predicted_prob"] = y_hat


    save_data(df, f"data/predicted/predicted_{model}_{experiment_name}.csv")
    print("Done")
    return
    

if __name__=="__main__":
    main()