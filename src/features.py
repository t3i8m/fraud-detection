import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y):
        X = X.copy()
        X['target'] = y.values
        self.card_fraud_history_ = X.groupby('card_id')['target'].sum()
        return self
    

    def _card_had_fraud(self, X, y):
        X = X.copy()
        X["target"] = y.values
        X = X.sort_values(["card_id", "date"])
        X["prev_fraud_count"] = (X.groupby("card_id")["target"].transform(lambda x: x.shift(1).fillna(0).cumsum()))
        X = X.drop(columns=["target"])
        return X


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

        X_train["is_online"] = (X_train["use_chip"].str.lower().str.contains("online"))
        X_train["has_bad_cvv"] = X_train["errors"].str.lower().str.contains("cvv")

        return X_train



class RareCategoryGrouper(BaseEstimator, TransformerMixin):

    def __init__(self, threshold=1000, columns=['merchant_city']):
        super().__init__()
        self.threshold=threshold
        self.columns=columns


    def fit(self, X, y=None):
        self.frequent_categories_ = {}

        for n in self.columns:
            categories = X[n].value_counts()
            categories= list(categories.loc[categories>=self.threshold].index)
            self.frequent_categories_[n]=categories

        return self
    

    def transform(self, X, y=None):
        X = X.copy()

        for n in self.columns:
            X[n] = X[n].map(lambda x: x if x in self.frequent_categories_[n] else 'Other')

        return X

