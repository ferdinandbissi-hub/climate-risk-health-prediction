"""Competition metric and threshold search."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from .config import AUC_WEIGHT, F1_WEIGHT

DEFAULT_THRESHOLDS = np.arange(0.05, 0.96, 0.01)


def competition_score(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Return F1, ROC-AUC and the weighted competition score at a given threshold."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    f1 = f1_score(y_true, (y_prob >= threshold).astype(int))
    auc = roc_auc_score(y_true, y_prob)
    return {
        "F1": f1,
        "ROC_AUC": auc,
        "Competition_Score": F1_WEIGHT * f1 + AUC_WEIGHT * auc,
    }


def find_best_threshold(y_true, y_prob, thresholds=DEFAULT_THRESHOLDS):
    """Scan thresholds and return (best_row, full_results_table).

    ROC-AUC does not depend on the threshold, so it is computed only once.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    auc = roc_auc_score(y_true, y_prob)

    rows = []
    for threshold in thresholds:
        f1 = f1_score(y_true, (y_prob >= threshold).astype(int))
        rows.append({
            "threshold": round(float(threshold), 2),
            "f1": f1,
            "roc_auc": auc,
            "score": F1_WEIGHT * f1 + AUC_WEIGHT * auc,
        })

    results = pd.DataFrame(rows)
    best = results.loc[results["score"].idxmax()]
    return best, results
