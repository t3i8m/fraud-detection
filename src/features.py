from category_encoders import CatBoostEncoder
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin



class FeatureTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, columns_to_drop=None):
        super().__init__()
        self.columns_to_drop_ = columns_to_drop if columns_to_drop is not None else []


    def fit(self, X, y,):
        X = X.copy()
        X["merchant_id"] = X["merchant_id"].astype(str)

        self._mcc_encoder = CatBoostEncoder(cols=['mcc'], a=10)
        self._merchant_encoder = CatBoostEncoder(cols=['merchant_id'], a=10)
        self._merchant_state_encoder = CatBoostEncoder(cols=['merchant_state'], a=10)

        self._mcc_encoder.fit(X["mcc"], y)
        self._merchant_encoder.fit(X["merchant_id"], y)
        self._merchant_state_encoder.fit(X["merchant_state"], y)

        return self

    def transform(self, X, y=None):
        X_train = X.copy()
        X_train["merchant_id"] = X_train["merchant_id"].astype(str)

        X_train["is_refund"] = (X_train["amount"]<0).astype(int)
        X_train['hour'] = X_train['date'].dt.hour
        X_train['day_of_week'] = X_train['date'].dt.dayofweek
        X_train['month'] = X_train['date'].dt.month

        X_train["is_midday"] = ((X_train['hour']>=10) & (X_train['hour']<=14))

        for col, default in [("prev_fraud_count", 0),("time_since_last_trx", -1),("card_swipe_ratio", 0),("is_new_mcc", 0),("user_amount_z_score", 0),]:
            if col not in X_train.columns:
                X_train[col] = default

        X_train["is_online"] = (X_train["use_chip"].str.lower().str.contains("online", na=False))
        # error columns
        X_train["errors"] = X_train["errors"].fillna("No error")

        X_train["mcc_risk"] = self._mcc_encoder.transform(X_train["mcc"], y)
        X_train["merchant_risk"] = self._merchant_encoder.transform(X_train["merchant_id"], y)
        X_train["merchant_state_risk"] = self._merchant_state_encoder.transform(X_train["merchant_state"], y)

        if self.columns_to_drop_:
            X_train = X_train.drop(columns=self.columns_to_drop_, axis = 1, errors='ignore')

        self.feature_names_out_ = list(X_train.columns)

        return X_train
    

    def get_feature_names_out(self, input_features=None):
        if hasattr(self, "feature_names_out_"):
            return np.array(self.feature_names_out_, dtype=object)
        return np.array(input_features) if input_features is not None else None
    
    
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X,y)
        return self.transform(X,y)



class RareCategoryGrouper(BaseEstimator, TransformerMixin):

    def __init__(self, threshold=1000, cat_columns=['merchant_city']):
        super().__init__()
        self.threshold=threshold
        self.cat_columns=cat_columns
        self.frequent_categories_ = {}


    def fit(self, X, y=None):
        self.frequent_categories_ = {}

        for n in self.cat_columns:
            categories = X[n].value_counts()
            categories= list(categories.loc[categories>=self.threshold].index)
            self.frequent_categories_[n]=categories

        return self
    

    def transform(self, X, y=None):
        X = X.copy()

        for n in self.cat_columns:
            X[n] = X[n].where(X[n].isin(self.frequent_categories_[n]), 'Other')

        self.feature_names_out_ = list(X.columns)
        return X
    

    def get_feature_names_out(self, input_features=None):
        if hasattr(self, "feature_names_out_"):
            return np.array(self.feature_names_out_, dtype=object)
        return np.array(input_features) if input_features is not None else None

