"""Simple CSV experiment tracker (EXP-001, EXP-002, ...)."""
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import EXPERIMENT_LOG

COLUMNS = [
    "exp_id", "model", "feature_set", "hyperparameters", "validation",
    "threshold", "F1", "ROC_AUC", "score", "score_at_0.5",
    "fold_auc_mean", "fold_auc_std", "conclusion",
]


class ExperimentLogger:
    """Append one row per experiment and keep it on disk."""

    def __init__(self, path: Path = EXPERIMENT_LOG):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def reset(self) -> None:
        """Delete the existing log (use at the top of a notebook for a clean run)."""
        if self.path.exists():
            self.path.unlink()

    def load(self) -> pd.DataFrame:
        if self.path.exists():
            return pd.read_csv(self.path)
        return pd.DataFrame(columns=COLUMNS)

    def log(
        self,
        model: str,
        feature_set: str,
        hyperparameters: Optional[dict],
        validation: str,
        metrics: dict,
        conclusion: str = "",
    ) -> str:
        """Record an experiment and return its ID."""
        history = self.load()
        exp_id = f"EXP-{len(history) + 1:03d}"
        row = {
            "exp_id": exp_id,
            "model": model,
            "feature_set": feature_set,
            "hyperparameters": str(hyperparameters or {}),
            "validation": validation,
            "threshold": metrics["threshold"],
            "F1": round(metrics["F1"], 4),
            "ROC_AUC": round(metrics["ROC_AUC"], 4),
            "score": round(metrics["score"], 4),
            "score_at_0.5": round(metrics["score_at_0.5"], 4),
            "fold_auc_mean": round(metrics["fold_auc_mean"], 4),
            "fold_auc_std": round(metrics["fold_auc_std"], 4),
            "conclusion": conclusion,
        }
        updated = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
        updated.to_csv(self.path, index=False)
        return exp_id
