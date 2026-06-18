
from src.data_loader import load_data
from src.train import train_timeseries_cv


def main():
    model = "xgboost"

    df = load_data("data/processed/transactions_train.csv")
    print("Data was loaded", df.shape)
    _,_ = train_timeseries_cv(df, model)
    print("Done")
    return
    


if __name__=="__main__":
    main()