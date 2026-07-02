from category_encoders import TargetEncoder
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin



class FeatureTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, columns_to_drop=None):
        super().__init__()
        self.columns_to_drop_ = columns_to_drop if columns_to_drop is not None else []
        # self.card_fraud_history_ = None


    def fit(self, X, y,):
        X = X.copy()
        X['target'] = y.values
        self._mcc_encoder = TargetEncoder(cols=['mcc'], smoothing=10)
        self._mcc_encoder.fit(X["mcc"], y)
        # self.card_fraud_history_ = X.groupby('card_id')['target'].sum()
        return self
    

    # def _card_had_fraud(self, X, y):
    #     X = X.copy()
    #     orig_index = X.index
    #     X["target"] = y.values
    #     X = X.sort_values(["card_id", "date"])
    #     X["prev_fraud_count"] = (X.groupby("card_id")["target"].transform(lambda x: x.shift(1).fillna(0).cumsum()))
    #     X = X.loc[orig_index]
    #     X = X.drop(columns=["target"])
    #     return X

    
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

        if "prev_fraud_count" in X_train.columns:
          pass 
        else:
            X_train['prev_fraud_count'] = 0 

        X_train["is_online"] = (X_train["use_chip"].str.lower().str.contains("online", na=False))

        # error columns
        X_train["errors"] = X_train["errors"].fillna("No error")

        X_train["mcc"]= self._mcc_encoder.transform(X_train["mcc"])
        
        # X_train["mcc_category"] = self._map_mcc_to_category(X_train)
        X_train["time_since_last_trx"] = (X_train.groupby("card_id")["date"].diff().dt.total_seconds() .fillna(-1).astype(float))


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

