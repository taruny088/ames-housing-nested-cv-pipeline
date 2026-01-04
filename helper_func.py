import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

import category_encoders as ce






def rmsle(y_true, y_pred):
    p = np.maximum(y_pred, 0)
    a = np.maximum(y_true, 0)
    return float(np.sqrt(np.mean((np.log1p(p) - np.log1p(a))**2)))


class EnsureDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        # IMPORTANT: store exactly as passed (no list(), no copy)
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        # Convert to list only at transform-time (this is OK)
        cols = tuple(self.columns) if self.columns is not None else None
        return pd.DataFrame(X, columns=cols)


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """
    Groups rare categories and LOGS categories
    whose frequency is below min_freq.
    """
    def __init__(self, min_freq=0.01):
        self.min_freq = min_freq

    def fit(self, X, y=None):
        X = X.copy()
        n = len(X)

        self.keep_maps_ = {}
        self.rare_categories_ = {}   

        for col in X.columns:
            vc = X[col].astype("object").value_counts(dropna=False)

            keep = vc[vc / n >= self.min_freq].index.tolist()
            rare = vc[vc / n < self.min_freq].index.tolist()  

            self.keep_maps_[col] = set(keep)
            self.rare_categories_[col] = rare  

        return self

    def transform(self, X):
        X = X.copy()
        for col in X.columns:
            keep = self.keep_maps_.get(col)
            X[col] = X[col].astype("object").where(X[col].isin(keep), "__RARE__")
        return X




def _nearest_odd(k, min_k=3):
    """
    Helper to round a value to the nearest odd integer,
    enforcing a minimum value.
    """
    k = int(round(k))
    if k % 2 == 0:
        k += 1
    return max(k, min_k)


def build_preprocessor(params, num_cols, cat_cols):
    #dynamic KNN neighbors (rounded to nearest odd)
    knn_k = _nearest_odd(params.get("knn_neighbors", 5))

    num_pipe = Pipeline([
        #was n_neighbors=5 (hardcoded)
        ("imp", KNNImputer(n_neighbors=knn_k)),

        ("poly", PolynomialFeatures(
            degree=2,
            interaction_only=True,
            include_bias=False
        )),
        ("scale", RobustScaler()),
        ("pca", PCA(n_components=params["pca_var"], svd_solver="full")),
    ])

    cat_cols_tuple = tuple(cat_cols)
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("to_df", EnsureDataFrame(cat_cols_tuple)),
        ("rare", RareCategoryGrouper(min_freq=params["rare_min_freq"])),
        ("loo", ce.LeaveOneOutEncoder(
            sigma=params["loo_smooth"],
            handle_unknown="value",
            handle_missing="value"
        )),
    ])

    return ColumnTransformer(transformers=[("num", num_pipe, num_cols),("cat", cat_pipe, cat_cols),],remainder="drop",verbose_feature_names_out=True)




def build_model(params, random_state):
    if params["model_type"] == "XGB":
        return xgb.XGBRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["xgb_max_depth"],
            subsample=params["xgb_subsample"],
            colsample_bytree=params["xgb_colsample"],
            reg_lambda=params["xgb_reg_lambda"],
            min_child_weight=params["xgb_min_child_weight"],
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
        )
    else:
        return lgb.LGBMRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            num_leaves=params["lgb_num_leaves"],
            min_child_samples=params["lgb_min_child_samples"],
            subsample=params["lgb_subsample"],
            colsample_bytree=params["lgb_colsample"],
            reg_lambda=params["lgb_reg_lambda"],
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )



def suggest_params(trial):
    model_type = trial.suggest_categorical("model_type", ["XGB", "LGBM"])
    params = {
        "model_type": model_type,
        "pca_var": trial.suggest_float("pca_var", 0.85, 0.98),
        "loo_smooth": trial.suggest_float("loo_smooth", 1.0, 5.0),
        "rare_min_freq": trial.suggest_float("rare_min_freq", 0.005, 0.03),
        "n_estimators": trial.suggest_int("n_estimators", 600, 2400, step=600),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
    }

    if model_type == "XGB":
        params.update({
            "xgb_max_depth": trial.suggest_int("xgb_max_depth", 3, 8),
            "xgb_subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
            "xgb_colsample": trial.suggest_float("xgb_colsample", 0.6, 1.0),
            "xgb_reg_lambda": trial.suggest_float("xgb_reg_lambda", 1e-3, 10.0, log=True),
            "xgb_min_child_weight": trial.suggest_float("xgb_min_child_weight", 0.5, 10.0, log=True),
        })
    else:
        params.update({
            "lgb_num_leaves": trial.suggest_int("lgb_num_leaves", 20, 96),
            "lgb_min_child_samples": trial.suggest_int("lgb_min_child_samples", 5, 60),
            "lgb_subsample": trial.suggest_float("lgb_subsample", 0.6, 1.0),
            "lgb_colsample": trial.suggest_float("lgb_colsample", 0.6, 1.0),
            "lgb_reg_lambda": trial.suggest_float("lgb_reg_lambda", 1e-3, 10.0, log=True),
        })

    return params



def train_with_es_get_best_iter(X_train,y_train,params,num_cols,cat_cols,random_state,es_split,early_stopping_rounds,):
    """
    Fits preprocessor on train_sub only (strict),
    runs ES on es_split, returns best iteration + trained pre.
    """
    iso = IsolationForest(contamination=0.01, random_state=random_state)
    mask = iso.fit_predict(
        X_train[num_cols].fillna(
            X_train[num_cols].median(numeric_only=True)
        )
    )
    X_train = X_train.loc[mask == 1].copy()
    y_train = y_train.loc[mask == 1].copy()

    X_tr_sub, X_es, y_tr_sub, y_es = train_test_split(
        X_train,
        y_train,
        test_size=es_split,
        random_state=random_state
    )

    pre = build_preprocessor(params, num_cols, cat_cols)
    model = build_model(params, random_state)

    X_tr_t = pre.fit_transform(X_tr_sub, y_tr_sub)
    y_tr_t = np.log1p(y_tr_sub)

    X_es_t = pre.transform(X_es)
    y_es_t = np.log1p(y_es)

    if params["model_type"] == "XGB":
        model.fit(
            X_tr_t,
            y_tr_t,
            eval_set=[(X_es_t, y_es_t)],
            verbose=False,
        )
        best_iter = int(getattr(model, "best_iteration", model.n_estimators))
    else:
        model.fit(
            X_tr_t,
            y_tr_t,
            eval_set=[(X_es_t, y_es_t)],
            eval_metric="rmse",
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=early_stopping_rounds,
                    verbose=False
                )
            ],
        )
        best_iter = int(getattr(model, "best_iteration_", model.n_estimators))

    if best_iter <= 0:
        best_iter = model.n_estimators

    return pre, best_iter




def refit_on_full_training(
    X_train,
    y_train,
    params,
    num_cols,
    cat_cols,
    random_state,
    es_split,
    early_stopping_rounds,
):
    """
    Office-grade refit:
    1) ES to determine best_iter (train-only)
    2) Fit preprocessor on FULL training (after anomaly filter)
    3) Fit model on FULL transformed training using best_iter
    """
    iso = IsolationForest(contamination=0.01, random_state=random_state)
    mask = iso.fit_predict(
        X_train[num_cols].fillna(
            X_train[num_cols].median(numeric_only=True)
        )
    )
    X_train_f = X_train.loc[mask == 1].copy()
    y_train_f = y_train.loc[mask == 1].copy()

    pre_es, best_iter = train_with_es_get_best_iter(
        X_train=X_train_f,
        y_train=y_train_f,
        params=params,
        num_cols=num_cols,
        cat_cols=cat_cols,
        random_state=random_state,
        es_split=es_split,
        early_stopping_rounds=early_stopping_rounds,
    )

    pre = build_preprocessor(params, num_cols, cat_cols)
    X_tr_full_t = pre.fit_transform(X_train_f, y_train_f)
    y_tr_full_t = np.log1p(y_train_f)

    params2 = dict(params)
    params2["n_estimators"] = best_iter

    model = build_model(params2, random_state)
    model.fit(X_tr_full_t, y_tr_full_t)

    return {"pre": pre,"model": model,"params": params2,"best_iter": best_iter,}



def predict_artifact(artifact, X):
    X_t = artifact["pre"].transform(X)
    pred_log = artifact["model"].predict(X_t)
    return np.expm1(pred_log)


