from typing import Literal
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer

from src.features import FeatureTransformer, RareCategoryGrouper

RANDOM_STATE = 42


def build_pipeline(model_type:Literal["lgbm", "xgboost", "rf", "lr", "catboost"]= "xgboost", cols_to_drop=[], model_params=None) ->Pipeline:

    model_params = model_params or {}
    cols_to_drop = cols_to_drop or []

    cat_cols = ['mcc_category']
    num_cols = ['amount', 'hour', 'day_of_week', 'month', 'prev_fraud_count']

    cat_pipeline = Pipeline([("encoding", OneHotEncoder(handle_unknown="ignore"))])
    num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),("scaler", StandardScaler())])

    preprocessor = ColumnTransformer([("num_pipeline",num_pipeline, num_cols), ("cat_pipeline", cat_pipeline, cat_cols)], remainder="passthrough")

    if model_type=='lgbm':
        model = LGBMClassifier(scale_pos_weight=666, random_state=RANDOM_STATE, **model_params)
    elif model_type=='xgboost':
        model = XGBClassifier(scale_pos_weight=666, random_state=RANDOM_STATE, **model_params)
    elif model_type=="rf":
        model = RandomForestClassifier(random_state=RANDOM_STATE,**model_params)
    elif model_type=="rf":
        model = CatBoostClassifier(random_state=RANDOM_STATE,**model_params)
    else: 
        model = LogisticRegression(random_state=RANDOM_STATE,**model_params)
    
    return Pipeline([("feature_engineering", FeatureTransformer(columns_to_drop=cols_to_drop)), ("rare_categories", RareCategoryGrouper(cat_columns=cat_cols)), ("preprocessor", preprocessor), ("classifier", model)])


