
from config import THRESHOLD_PHYSICAL, THRESHOLD_ONLINE
from src.data_loader import load_data, save_data
from src.train import train_timeseries_cv
import pandas as pd


def main():
    model = "xgboost"
    online_trx = False
    exp_type = "ONLINE" if online_trx else "PHYSICAL"
    thr = THRESHOLD_ONLINE if online_trx else THRESHOLD_PHYSICAL
    experiment_name = f"{exp_type}_regularization_scale_pos_2860_added_fraud_count_maxdepth4_estimators150_added_added_behaviour_isnewmcc_merchant_risk10_cardswiperatio_merchantstaterisk_distanceadded_fraud_30d"
    mode_params = {}

    df = load_data("data/processed/transactions_train.csv",online_trx, users_path="data/raw/users_data.csv" , cards_path = "data/raw/cards_data.csv")
    
    df = df.sort_values("date").reset_index(drop=True)
    print("Data was loaded", df.shape)
    print(df["target"].value_counts(normalize=True))

    if model in ["xgboost", "catboost", "lightgbm"]:
        negative_count = df[df["target"]=="No"].shape[0]
        positive_count =  df[df["target"]=="Yes"].shape[0]
        scale_pos =  round(negative_count/positive_count)
        print("Scale pos:",scale_pos)
        mode_params["scale_pos_weight"]=scale_pos
        # for online 666

    y_true, y_hat = train_timeseries_cv(df, model, exp_type, experiment_name,online=online_trx,threshold=thr, model_params=mode_params)
    df["predicted_prob"] = y_hat


    save_data(df, f"data/predicted/train_predicted_{model}_{experiment_name}.csv")
    print("Done")
    return
    

if __name__=="__main__":
    main()