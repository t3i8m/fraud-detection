import pandas as pd


def extract_features(df):
    df = df.copy()
    if "target" in df.columns:
        df["temp_numeric_target"] = (df["target"].astype(str).str.lower().str.strip() == "yes").astype(int)
        df["prev_fraud_count"] = (df.groupby("card_id")["temp_numeric_target"].transform(lambda x: x.shift(1).fillna(0).cumsum()).astype(int))
        df = df.drop(columns = ["temp_numeric_target"])

    df["temp_is_online"] = df["use_chip"].str.lower().str.contains("online", na=False).astype(int)
    df["temp_prev_online_count"] = (df.groupby("card_id")["temp_is_online"].transform(lambda x: x.shift(1).fillna(0).cumsum()))
    df["temp_total_prev_count"] = df.groupby("card_id").cumcount()
    df["online_history_ratio"] = (df["temp_prev_online_count"] / df["temp_total_prev_count"]).fillna(0.0)

    df = df.set_index("date")
    df["trx_count_1h"] = (df.groupby("card_id")["amount"].transform(lambda x: x.rolling("1h").count() - 1).fillna(0))
    df["trx_amount_1h"] = (df.groupby("card_id")["amount"].transform(lambda x: x.rolling("1h").sum() - x).fillna(0))

    df["has_bad_cvv"] = df["errors"].str.contains("cvv", na=False).astype(int)
    df["has_bad_pin"] = df["errors"].str.contains("pin", na=False).astype(int)
    df["has_insufficient_balance"] = df["errors"].str.contains("balance", na=False).astype(int)
    df["has_technical_glitch"] = df["errors"].str.contains("glitch", na=False).astype(int)


    df["insufficient_balance_count_1h"] = (df.groupby("card_id")["has_insufficient_balance"].transform(lambda x: x.rolling("1h").sum() - x).fillna(0))
    df["bad_cvv_count_1h"] = (df.groupby("card_id")["has_bad_cvv"].transform(lambda x: x.rolling("1h").sum() - x).fillna(0))
    df["bad_pin_count_1h"] = (df.groupby("card_id")["has_bad_pin"].transform(lambda x: x.rolling("1h").sum() - x).fillna(0))
    df["tech_glitch_count_1h"] = (df.groupby("card_id")["has_technical_glitch"].transform(lambda x: x.rolling("1h").sum() - x).fillna(0))

    df = df.reset_index()
    
    df = df.drop(columns=["temp_is_online", "temp_prev_online_count", "temp_total_prev_count"])
    return df


def get_same_balance(df, subset):

    is_fraud = df["target"].astype(str).str.lower().str.strip().isin(["yes", "1", "1.0"])
    fraud = df[is_fraud]
    legit = df[~is_fraud]
    
    fraud_ratio = len(fraud) / len(df)
    
    n_fraud = int(subset * fraud_ratio)
    n_legit = subset - n_fraud
    
    fraud_sampled = fraud.sample(n=min(n_fraud, len(fraud)), random_state=42)
    legit_sampled = legit.sample(n=min(n_legit, len(legit)), random_state=42)
    
    df = pd.concat([fraud_sampled, legit_sampled]).sort_values("date").reset_index(drop=True)
    return df


def load_data(path: str, online_trx = False, subset: int = None):
    df = pd.read_csv(path, parse_dates=["date"])
    df = extract_features(df)

    if online_trx:
        condition = (df["use_chip"]=="Online Transaction")
    else:
        condition = (df["use_chip"]!="Online Transaction")

    df = df.loc[condition]
    
    if subset: 
        df = get_same_balance(df, subset)
        return df
    return df


def save_data(df:pd.DataFrame, path:str):
    df.to_csv(path, index=False)
    return


def split_by_data(df, split_date):
    train = df[df['date'] < split_date]
    test = df[df['date'] >= split_date]
    return train, test