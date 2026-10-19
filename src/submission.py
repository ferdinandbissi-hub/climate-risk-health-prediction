"""Build and check the competition submission file."""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SUBMISSION_COLUMNS, SUBMISSION_DIR


def make_submission(ids, test_prob, threshold: float, filename: str = "submission_v1.csv"):
    """Create the submission dataframe, run sanity checks and save it.

    Columns: ID, TargetF1 (0/1 label), TargetRAUC (probability).
    """
    test_prob = np.asarray(test_prob, dtype=float)

    submission = pd.DataFrame({
        SUBMISSION_COLUMNS["id"]: np.asarray(ids),
        SUBMISSION_COLUMNS["label"]: (test_prob >= threshold).astype(int),
        SUBMISSION_COLUMNS["proba"]: test_prob,
    })

    assert submission[SUBMISSION_COLUMNS["id"]].is_unique, "Duplicated IDs"
    assert not submission.isna().any().any(), "Missing values in submission"
    assert submission[SUBMISSION_COLUMNS["proba"]].between(0, 1).all(), "Probabilities outside [0, 1]"

    path = Path(SUBMISSION_DIR) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
    return submission, path
