
from config import THRESHOLD_PHYSICAL, THRESHOLD_ONLINE
from src.data_loader import load_data, save_data
from src.train import train_timeseries_cv
import pandas as pd

def main():
    model = "xgboost"
    online_trx = True
    exp_type = "ONLINE" if online_trx else "PHYSICAL"
    thr = THRESHOLD_ONLINE if online_trx else THRESHOLD_PHYSICAL
    experiment_name = f"{exp_type}_regularization_subset_added_velocity_features_removed_online_added_mcc_risk_score_velocity_errors"

    df = load_data("data/processed/transactions_train.csv",online_trx, subset = 2000000)
    print("Data was loaded", df.shape)
    print(df["target"].value_counts(normalize=True))

    y_true, y_hat = train_timeseries_cv(df, model, exp_type, experiment_name, threshold=thr)
    df["predicted_prob"] = y_hat

    save_data(df, f"data/predicted/train_predicted_{model}_{experiment_name}.csv")
    print("Done")
    return
    

if __name__=="__main__":
    main()