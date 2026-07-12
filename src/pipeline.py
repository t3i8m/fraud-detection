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


def build_pipeline(model_type:Literal["lgbm", "xgboost", "rf", "lr", "catboost"]= "xgboost", cols_to_drop=[], categoric_cols=[], numeric_cols=[], model_params=None) ->Pipeline:

    model_params = model_params or {}
    cols_to_drop = cols_to_drop or []

    cat_pipeline = Pipeline([("encoding", OneHotEncoder(handle_unknown="ignore"))])
    num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),("scaler", StandardScaler())])

    preprocessor = ColumnTransformer([("num_pipeline",num_pipeline, numeric_cols), ("cat_pipeline", cat_pipeline, categoric_cols)], remainder="passthrough")

    if model_type=='lgbm':
        model = LGBMClassifier(verbose=-1, max_depth=3,num_leaves=7,learning_rate=0.05,n_estimators=150,subsample=0.8,colsample_bytree=0.8,reg_lambda=10.0, random_state=RANDOM_STATE, **model_params)
    elif model_type=='xgboost':
        model_params.setdefault("max_depth", 4)
        model_params.setdefault("learning_rate", 0.05)
        model_params.setdefault("n_estimators", 300)
        model_params.setdefault("subsample", 0.8)
        model_params.setdefault("colsample_bytree", 0.8)
        model_params.setdefault("reg_lambda", 10.0)
        model_params.setdefault("reg_alpha", 5.0)
        model_params.setdefault("min_child_weight", 10)
        model_params.setdefault("gamma", 1.0)
        model = XGBClassifier(random_state=RANDOM_STATE, **model_params)
    elif model_type=="rf":
        model = RandomForestClassifier(max_depth=6, min_samples_leaf=20, n_estimators=100, class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE,**model_params)
    elif model_type=="catboost":
        model = CatBoostClassifier(depth=3, learning_rate=0.05,iterations=150, l2_leaf_reg=10.0, subsample=0.8, verbose=0, random_state=RANDOM_STATE,**model_params)
    else: 
        model = LogisticRegression(random_state=RANDOM_STATE, C=1.0,  max_iter=1000, **model_params)
    
    return Pipeline([("feature_engineering", FeatureTransformer(columns_to_drop=cols_to_drop)), ("rare_categories", RareCategoryGrouper(cat_columns=categoric_cols)), ("preprocessor", preprocessor), ("classifier", model)])


