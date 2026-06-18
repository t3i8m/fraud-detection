import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin



class FeatureTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, columns_to_drop=None):
        super().__init__()
        self.columns_to_drop_ = columns_to_drop if columns_to_drop is not None else []
        self.card_fraud_history_ = None


    def fit(self, X, y,):
        X = X.copy()
        X['target'] = y.values
        self.card_fraud_history_ = X.groupby('card_id')['target'].sum()
        return self
    

    def _card_had_fraud(self, X, y):
        X = X.copy()
        orig_index = X.index
        X["target"] = y.values
        X = X.sort_values(["card_id", "date"])
        X["prev_fraud_count"] = (X.groupby("card_id")["target"].transform(lambda x: x.shift(1).fillna(0).cumsum()))
        X = X.loc[orig_index]
        X = X.drop(columns=["target"])
        return X
    
    
    def _map_mcc_to_category(self, df):
        mcc = pd.to_numeric(df["mcc"], errors="coerce").fillna(-1).astype(int)
        conditions = [(mcc >= 3000) & (mcc <= 3299),(mcc >= 3300) & (mcc <= 3499),(mcc >= 3500) & (mcc <= 3999),(mcc >= 1) & (mcc <= 1499),(mcc >= 1500) & (mcc <= 2999),(mcc >= 4000) & (mcc <= 4799),(mcc >= 4800) & (mcc <= 4999),(mcc >= 5000) & (mcc <= 5599),(mcc >= 5600) & (mcc <= 5699),(mcc >= 5700) & (mcc <= 7299),(mcc >= 7300) & (mcc <= 7999),(mcc >= 8000) & (mcc <= 8999),(mcc >= 9000) & (mcc <= 9999),]
        choices = ['Airlines', 'Car_Rental', 'Hotels', 'Agriculture', 'Contracted_Services','Transportation', 'Utilities', 'Retail', 'Clothing', 'Misc_Shops','Business_Amusement', 'Professional_Services', 'Government']
        return np.select(conditions, choices, default='Other')


    def transform(self, X, y=None):
        X_train = X.copy()

        X_train["is_refund"] = (X_train["amount"]<0).astype(int)
        X_train['hour'] = X_train['date'].dt.hour
        X_train['day_of_week'] = X_train['date'].dt.dayofweek
        X_train['month'] = X_train['date'].dt.month

        X_train["is_midday"] = ((X_train['hour']>=10) & (X_train['hour']<=14))

        if y is None:
            X_train['prev_fraud_count'] = X_train['card_id'].map(self.card_fraud_history_).fillna(0)
        else:
            X_train = self._card_had_fraud(X_train,y)

        X_train["is_online"] = (X_train["use_chip"].str.lower().str.contains("online", na=False))
        X_train["has_bad_cvv"] = X_train["errors"].str.lower().str.contains("cvv", na=False)
        X_train["mcc_category"] = self._map_mcc_to_category(X_train)
        
        if X_train["amount"].dtype=="object":
            X_train["amount"] = X_train["amount"].str.replace("$","").astype(float)

        if self.columns_to_drop_:
            X_train = X_train.drop(columns=self.columns_to_drop_, axis = 1, errors='ignore')

        return X_train
    
    
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

        return X

