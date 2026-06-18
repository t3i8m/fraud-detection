import pandas as pd

def load_data(path:str):
    return pd.read_csv(path, parse_dates=["date"])


def split_by_data(df, split_date):
    train = df[df['date'] < split_date]
    test = df[df['date'] >= split_date]
    return train, test