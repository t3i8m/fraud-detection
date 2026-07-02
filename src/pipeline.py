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
from config import CATEGORIC_COLS, NUMERIC_COLS
from src.features import FeatureTransformer, RareCategoryGrouper

RANDOM_STATE = 42


def build_pipeline(model_type:Literal["lgbm", "xgboost", "rf", "lr", "catboost"]= "xgboost", cols_to_drop=[], model_params=None) ->Pipeline:

    model_params = model_params or {}
    cols_to_drop = cols_to_drop or []

    cat_pipeline = Pipeline([("encoding", OneHotEncoder(handle_unknown="ignore"))])
    num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),("scaler", StandardScaler())])

    preprocessor = ColumnTransformer([("num_pipeline",num_pipeline, NUMERIC_COLS), ("cat_pipeline", cat_pipeline, CATEGORIC_COLS)], remainder="passthrough")

    if model_type=='lgbm':
        model = LGBMClassifier(scale_pos_weight=666,verbose=-1, max_depth=3,num_leaves=7,learning_rate=0.05,n_estimators=150,subsample=0.8,colsample_bytree=0.8,reg_lambda=10.0, random_state=RANDOM_STATE, **model_params)
    elif model_type=='xgboost':
        model = XGBClassifier(scale_pos_weight=666, max_depth=3, learning_rate=0.05, n_estimators=150, subsample=0.8,colsample_bytree=0.8,reg_lambda=10.0, random_state=RANDOM_STATE, **model_params)
    elif model_type=="rf":
        model = RandomForestClassifier(max_depth=6, min_samples_leaf=20, n_estimators=100, class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE,**model_params)
    elif model_type=="catboost":
        model = CatBoostClassifier(depth=3, learning_rate=0.05,iterations=150, l2_leaf_reg=10.0, subsample=0.8,scale_pos_weight=666, verbose=0, random_state=RANDOM_STATE,**model_params)
    else: 
        model = LogisticRegression(random_state=RANDOM_STATE, C=1.0,  max_iter=1000, **model_params)
    
    return Pipeline([("feature_engineering", FeatureTransformer(columns_to_drop=cols_to_drop)), ("rare_categories", RareCategoryGrouper(cat_columns=CATEGORIC_COLS)), ("preprocessor", preprocessor), ("classifier", model)])


