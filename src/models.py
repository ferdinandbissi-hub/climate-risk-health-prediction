"""Model factory. Every model exposes fit / predict_proba."""
from typing import Callable, Dict, Optional

import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .config import RANDOM_STATE
from .features import get_categorical_features

MODEL_NAMES = ("logreg", "random_forest", "extra_trees", "lightgbm", "xgboost", "catboost")

DEFAULT_PARAMS: Dict[str, dict] = {
    "logreg": {"C": 1.0, "max_iter": 2000, "class_weight": "balanced"},
    "random_forest": {"n_estimators": 500, "min_samples_leaf": 5, "n_jobs": -1},
    "extra_trees": {"n_estimators": 500, "min_samples_leaf": 5, "n_jobs": -1},
    "lightgbm": {
        "n_estimators": 400, "learning_rate": 0.03, "num_leaves": 15,
        "min_child_samples": 20, "subsample": 0.8, "subsample_freq": 1,
        "colsample_bytree": 0.8, "reg_lambda": 5.0, "verbose": -1,
    },
    "xgboost": {
        "n_estimators": 400, "learning_rate": 0.03, "max_depth": 4,
        "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 5.0,
        "eval_metric": "logloss", "n_jobs": -1,
    },
    "catboost": {
        "iterations": 600, "learning_rate": 0.03, "depth": 6,
        "l2_leaf_reg": 5, "random_strength": 1,
    },
}


def _preprocessor(X: pd.DataFrame, scale: bool) -> ColumnTransformer:
    """Impute numeric columns, optionally scale them, one-hot encode categoricals."""
    cat_cols = get_categorical_features(X)
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers = [("num", Pipeline(numeric_steps), num_cols)]
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            cat_cols,
        ))
    return ColumnTransformer(transformers)


def make_builder(
    name: str,
    X: pd.DataFrame,
    params: Optional[dict] = None,
    seed: int = RANDOM_STATE,
) -> Callable:
    """Return a zero-argument function that builds a fresh, untrained model.

    `X` is only used to detect column types. `params` override the defaults.
    """
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name!r}. Choose from {MODEL_NAMES}")

    merged = {**DEFAULT_PARAMS[name], **(params or {})}
    cat_cols = get_categorical_features(X)

    def build():
        if name == "catboost":
            return CatBoostClassifier(
                loss_function="Logloss",
                random_seed=seed,
                verbose=False,
                allow_writing_files=False,
                cat_features=cat_cols,
                **merged,
            )

        if name == "logreg":
            estimator = LogisticRegression(random_state=seed, **merged)
            scale = True
        elif name == "random_forest":
            estimator = RandomForestClassifier(random_state=seed, **merged)
            scale = False
        elif name == "extra_trees":
            estimator = ExtraTreesClassifier(random_state=seed, **merged)
            scale = False
        elif name == "lightgbm":
            estimator = LGBMClassifier(random_state=seed, **merged)
            scale = False
        else:  # xgboost
            estimator = XGBClassifier(random_state=seed, **merged)
            scale = False

        return Pipeline([
            ("preprocessor", _preprocessor(X, scale=scale)),
            ("model", estimator),
        ])

    return build
