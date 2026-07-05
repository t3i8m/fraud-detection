import pandas as pd
import pgeocode
import numpy as np


def extract_features(df):
    df = df.copy()

    df = df.sort_values("date").reset_index(drop=True)

    df["temp_is_online"] = df["use_chip"].str.lower().str.contains("online", na=False).astype(int)
    df["temp_prev_online_count"] = (df.groupby("card_id")["temp_is_online"].transform(lambda x: x.shift(1).fillna(0).cumsum()))
    df["temp_total_prev_count"] = df.groupby("card_id").cumcount()
    df["online_history_ratio"] = (df["temp_prev_online_count"] / df["temp_total_prev_count"]).fillna(0.0)

    df["time_since_last_trx"] = (df.groupby("card_id")["date"].diff().dt.total_seconds().fillna(-1).astype(float))

    is_swipe = (df["use_chip"] == "Swipe Transaction").astype(int)
    df["card_swipe_ratio"] = is_swipe.groupby(df["card_id"]).transform(lambda s: s.shift(1).expanding().mean().fillna(0))

    df["is_new_mcc"] = (~df.duplicated(subset=["client_id", "mcc"])).astype(int)
    df["is_new_merchant"] = (~df.duplicated(subset=["client_id", "merchant_id"])).astype(int)

    grouped_amount = df.groupby("client_id")["amount"]
    exp_mean = grouped_amount.transform(lambda x: x.shift(1).expanding().mean().fillna(0))
    exp_std = grouped_amount.transform(lambda x: x.shift(1).expanding().std().fillna(0))
    df["user_amount_z_score"] = (df["amount"] - exp_mean) / exp_std.replace(0, 1.0)

    df = df.set_index("date")
    df["trx_count_1h"] = (df.groupby("card_id")["amount"].transform(lambda x: x.rolling("1h").count() - 1).fillna(0))
    df["trx_amount_1h"] = (df.groupby("card_id")["amount"].transform(lambda x: x.rolling("1h").sum() - x).fillna(0))

    if "target" in df.columns:
        df["temp_numeric_target"] = (df["target"].astype(str).str.lower().str.strip() == "yes").astype(int)
        df["prev_fraud_count"] = (df.groupby("card_id")["temp_numeric_target"].transform(lambda x: x.rolling("30D").sum() - x).fillna(0).astype(int))
        df = df.drop(columns=["temp_numeric_target"])

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


def extract_features_offline(df):
    nomi = pgeocode.Nominatim('us')
    
    clean_zips = pd.to_numeric(df["zip"], errors="coerce").fillna(0).astype(int).astype(str).str.zfill(5)
    clean_zips = clean_zips.replace("00000", np.nan)
    
    geo_df = nomi.query_postal_code(clean_zips.values)

    df["transaction_latitude"] = geo_df["latitude"].values
    df["transaction_longitude"] = geo_df["longitude"].values

    df["distance_to_home_km"] = haversine_vectorized(df["longitude"], df["latitude"], df["transaction_longitude"], df["transaction_latitude"])
    df["location_type"] = np.where(df["distance_to_home_km"] > 100, "Far (>100km)", "Close (<100km)")

    df["temp_orig_index"] = df.index
    df_sorted = df.sort_values(["card_id", "date"])
    
    df_sorted["prev_lat"] = df_sorted.groupby("card_id")["transaction_latitude"].shift(1)
    df_sorted["prev_lon"] = df_sorted.groupby("card_id")["transaction_longitude"].shift(1)
    
    df_sorted["dist_from_prev_km"] = haversine_vectorized(df_sorted["prev_lon"], df_sorted["prev_lat"],df_sorted["transaction_longitude"], df_sorted["transaction_latitude"])
    time_diff_hours = df_sorted.groupby("card_id")["date"].diff().dt.total_seconds() / 3600.0
    
    df_sorted["velocity_km_h"] = df_sorted["dist_from_prev_km"] / time_diff_hours.replace(0, np.nan)
    df_sorted["velocity_km_h"] = df_sorted["velocity_km_h"].fillna(0)
    
    df = df_sorted.sort_values("temp_orig_index").drop(columns=["temp_orig_index", "prev_lat", "prev_lon", "dist_from_prev_km"])
    df = df.drop(columns=["transaction_longitude", "transaction_latitude"])

    grouped_dist = df.groupby("client_id")["distance_to_home_km"]
    exp_mean_dist = grouped_dist.transform(lambda x: x.shift(1).expanding().mean().fillna(0))
    exp_std_dist = grouped_dist.transform(lambda x: x.shift(1).expanding().std().fillna(0))
    df["distance_z_score"] = (df["distance_to_home_km"] - exp_mean_dist) / exp_std_dist.replace(0, 1.0)

    return df



def get_same_balance(df, subset):

    is_fraud = (df["target"]=="Yes")
    fraud = df[is_fraud]
    legit = df[~is_fraud]
    
    fraud_ratio = len(fraud) / len(df)
    
    n_fraud = int(subset * fraud_ratio)
    n_legit = subset - n_fraud
    
    fraud_sampled = fraud.sample(n=min(n_fraud, len(fraud)), random_state=42)
    legit_sampled = legit.sample(n=min(n_legit, len(legit)), random_state=42)
    
    df = pd.concat([fraud_sampled, legit_sampled]).sort_values("date").reset_index(drop=True)
    return df


def undersampling_df(df, target_ratio=0.01):
    is_fraud = df["target"].astype(str).str.lower().str.strip().isin(["yes", "1", "1.0"])
    fraud = df[is_fraud]
    legit = df[~is_fraud]
    
    n_fraud = len(fraud) 
    n_legit = int(n_fraud / target_ratio) - n_fraud
    
    legit_sampled = legit.sample(n=min(n_legit, len(legit)), random_state=42)
    
    df_balanced = pd.concat([fraud, legit_sampled]).sort_values("date").reset_index(drop=True)
    return df_balanced


def load_data(path: str, online_trx = False, users_path:str=None,cards_path:str=None, undersampling = False, subset: int = None):
    df = pd.read_csv(path, parse_dates=["date"])
    df = extract_features(df)

    if online_trx:
        condition = (df["use_chip"]=="Online Transaction")
    else:
        users_data = pd.read_csv(users_path).rename(columns={"id":"client_id"})
        cards_data = pd.read_csv(cards_path).rename(columns={"id":"card_id"}).drop(columns=["client_id"])

        df = df.merge(users_data, on="client_id")
        df = df.merge(cards_data, on=["card_id"])
        df = extract_features_offline(df)
        condition = (df["use_chip"]!="Online Transaction")

    df = df.loc[condition]

    if subset: 
        df = get_same_balance(df, subset)
    elif undersampling:
        df = undersampling_df(df)

    return df



def haversine_vectorized(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c 
    return km


def save_data(df:pd.DataFrame, path:str):
    df.to_csv(path, index=False)
    return


def split_by_data(df, split_date):
    train = df[df['date'] < split_date]
    test = df[df['date'] >= split_date]
    return train, test