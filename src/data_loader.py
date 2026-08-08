from collections import defaultdict, deque
import pandas as pd
import pgeocode
import numpy as np


def extract_features(df):
    df = df.copy()

    df["amount"] = df["amount"].astype(str).str.replace("$", "", regex=False).astype(float)

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

    if "target" in df.columns:
        is_fraud = (df["target"].astype(str).str.lower().str.strip() == "yes")
        legit_amount = df["amount"].where(~is_fraud)
    else:
        legit_amount = df["amount"]

    grouped_legit_amount = legit_amount.groupby(df["client_id"])
    exp_mean = grouped_legit_amount.transform(lambda x: x.shift(1).expanding().mean().fillna(0))
    exp_std = grouped_legit_amount.transform(lambda x: x.shift(1).expanding().std().fillna(0))
    df["user_amount_z_score"] = (df["amount"] - exp_mean) / exp_std.replace(0, 1.0)

    df = df.set_index("date")
    df["trx_count_1h"] = (df.groupby("card_id")["amount"].transform(lambda x: x.rolling("1h").count() - 1).fillna(0))
    df["trx_amount_1h"] = (df.groupby("card_id")["amount"].transform(lambda x: x.rolling("1h").sum() - x).fillna(0))

    if "target" in df.columns:
        df["temp_numeric_target"] = (df["target"].astype(str).str.lower().str.strip() == "yes").astype(int)
        df["prev_fraud_count_30days"] = (df.groupby("card_id")["temp_numeric_target"].transform(lambda x: x.rolling("30D").sum() - x).fillna(0).astype(int))
        df = df.drop(columns=["temp_numeric_target"])

    df["has_bad_cvv"] = df["errors"].str.contains("cvv", case=False, na=False).astype(int)
    df["has_bad_pin"] = df["errors"].str.contains("pin", case=False, na=False).astype(int)
    df["has_insufficient_balance"] = df["errors"].str.contains("balance", case=False, na=False).astype(int)
    df["has_technical_glitch"] = df["errors"].str.contains("glitch", case=False, na=False).astype(int)

    df["insufficient_balance_count_1h"] = (df.groupby("card_id")["has_insufficient_balance"].transform(lambda x: x.rolling("1h").sum() - x).fillna(0))
    df["bad_cvv_count_1h"] = (df.groupby("card_id")["has_bad_cvv"].transform(lambda x: x.rolling("1h").sum() - x).fillna(0))
    df["bad_pin_count_1h"] = (df.groupby("card_id")["has_bad_pin"].transform(lambda x: x.rolling("1h").sum() - x).fillna(0))
    df["tech_glitch_count_1h"] = (df.groupby("card_id")["has_technical_glitch"].transform(lambda x: x.rolling("1h").sum() - x).fillna(0))

    df = df.reset_index()
    
    df = df.drop(columns=["temp_is_online", "temp_prev_online_count", "temp_total_prev_count"])
    return df


def add_lagged_client_features(df, channel_mask, lag_days=14):
    df = df.copy()
    fraud_flags = (df["target"].astype(str).str.lower().str.strip() == "yes").values if "target" in df.columns else np.zeros(len(df), dtype=bool)

    dates = df["date"].values
    client_ids = df["client_id"].values
    mccs = df["mcc"].values
    merchants = df["merchant_id"].values
    amounts = df["amount"].values

    lag = np.timedelta64(lag_days, "D")

    pending_mcc = defaultdict(deque)
    pending_merchant = defaultdict(deque)
    pending_amount = defaultdict(deque)
    seen_mcc = defaultdict(set)
    seen_merchant = defaultdict(set)
    running_sum = {}
    running_sumsq = {}
    running_count = {}

    is_new_mcc = df["is_new_mcc"].values.copy()
    is_new_merchant = df["is_new_merchant"].values.copy()
    z = df["user_amount_z_score"].values.copy()

    for i in range(len(df)):
        if not channel_mask[i]:
            continue

        c = client_ids[i]
        t = dates[i]
        cutoff = t - lag

        q = pending_mcc[c]
        while q and q[0][0] <= cutoff:
            _, mcc_val, fraud_flag = q.popleft()
            if not fraud_flag:
                seen_mcc[c].add(mcc_val)

        q = pending_merchant[c]
        while q and q[0][0] <= cutoff:
            _, merch_val, fraud_flag = q.popleft()
            if not fraud_flag:
                seen_merchant[c].add(merch_val)

        q = pending_amount[c]
        while q and q[0][0] <= cutoff:
            _, amt_val, fraud_flag = q.popleft()
            if not fraud_flag:
                running_sum[c] = running_sum.get(c, 0.0) + amt_val
                running_sumsq[c] = running_sumsq.get(c, 0.0) + amt_val ** 2
                running_count[c] = running_count.get(c, 0) + 1

        is_new_mcc[i] = 0 if mccs[i] in seen_mcc[c] else 1
        is_new_merchant[i] = 0 if merchants[i] in seen_merchant[c] else 1

        n = running_count.get(c, 0)
        if n == 0:
            mean, std = 0.0, 1.0
        else:
            mean = running_sum[c] / n
            var = running_sumsq[c] / n - mean ** 2
            std = var ** 0.5 if var > 0 else 1.0
        z[i] = (amounts[i] - mean) / (std if std != 0 else 1.0)

        pending_mcc[c].append((t, mccs[i], fraud_flags[i]))
        pending_merchant[c].append((t, merchants[i], fraud_flags[i]))
        pending_amount[c].append((t, amounts[i], fraud_flags[i]))

    df["is_new_mcc"] = is_new_mcc
    df["is_new_merchant"] = is_new_merchant
    df["user_amount_z_score"] = z
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


def load_data(df, online_trx = False, users_path:str=None,cards_path:str=None, undersampling = False, subset: int = None):
    df = extract_features(df)

    if online_trx:
        condition = (df["use_chip"]=="Online Transaction")
        df = add_lagged_client_features(df, condition.values, lag_days=14)
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

    if online_trx:
        train_df = df[df['date'].dt.year.between(2010, 2015)]
        test_df = df[df['date'].dt.year == 2016]
    else:
        train_df = df[df['date'].dt.year == 2018]
        test_df = df[df['date'].dt.year == 2019]

    return train_df, test_df



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