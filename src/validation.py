"""Validation strategies and out-of-fold evaluation."""
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, TimeSeriesSplit

from .config import N_FOLDS, RANDOM_STATE
from .metrics import competition_score, find_best_threshold

Folds = List[Tuple[np.ndarray, np.ndarray]]

STRATEGIES = ("stratified", "group", "time")


def make_folds(
    y: pd.Series,
    strategy: str = "group",
    groups: Optional[pd.Series] = None,
    dates: Optional[pd.Series] = None,
    n_splits: int = N_FOLDS,
    seed: int = RANDOM_STATE,
) -> Folds:
    """Create (train_idx, valid_idx) pairs.

    strategy:
      - "stratified": random folds keeping the class ratio.
      - "group": no group (location) is shared between train and validation,
        class ratio kept as well as possible. Mimics the test set, where almost
        every location is new.
      - "time": expanding window; each fold validates on a later period.
        The earliest block is never used for validation.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")

    X_dummy = np.zeros(len(y))

    if strategy == "stratified":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(X_dummy, y))

    if strategy == "group":
        if groups is None:
            raise ValueError("`groups` is required for the 'group' strategy")
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(X_dummy, y, groups))

    if dates is None:
        raise ValueError("`dates` is required for the 'time' strategy")
    order = np.argsort(pd.to_datetime(dates).to_numpy(), kind="stable")
    splitter = TimeSeriesSplit(n_splits=n_splits)
    return [(order[tr], order[va]) for tr, va in splitter.split(order)]


def cross_validate(
    build_model: Callable,
    X: pd.DataFrame,
    y: pd.Series,
    folds: Folds,
    X_test: Optional[pd.DataFrame] = None,
) -> Dict:
    """Train one fresh model per fold and collect out-of-fold predictions.

    Returns a dict with:
      oof        : out-of-fold probabilities (NaN for rows never validated)
      test_pred  : mean of the fold models' test probabilities (or None)
      fold_auc   : ROC-AUC of each fold
      covered    : boolean mask of rows that received an out-of-fold prediction
    """
    from sklearn.metrics import roc_auc_score

    oof = np.full(len(X), np.nan)
    test_preds = []
    fold_auc = []

    for train_idx, valid_idx in folds:
        model = build_model()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]
        fold_auc.append(roc_auc_score(y.iloc[valid_idx], oof[valid_idx]))
        if X_test is not None:
            test_preds.append(model.predict_proba(X_test)[:, 1])

    return {
        "oof": oof,
        "test_pred": np.mean(test_preds, axis=0) if test_preds else None,
        "fold_auc": fold_auc,
        "covered": ~np.isnan(oof),
    }


def evaluate_oof(y: pd.Series, result: Dict) -> Dict:
    """Score out-of-fold predictions with the best threshold and the default 0.5."""
    mask = result["covered"]
    y_cov = y[mask]
    oof_cov = result["oof"][mask]

    best, table = find_best_threshold(y_cov, oof_cov)
    at_half = competition_score(y_cov, oof_cov, threshold=0.5)
    return {
        "threshold": float(best["threshold"]),
        "F1": float(best["f1"]),
        "ROC_AUC": float(best["roc_auc"]),
        "score": float(best["score"]),
        "score_at_0.5": float(at_half["Competition_Score"]),
        "fold_auc_mean": float(np.mean(result["fold_auc"])),
        "fold_auc_std": float(np.std(result["fold_auc"])),
        "n_scored": int(mask.sum()),
        "table": table,
    }
