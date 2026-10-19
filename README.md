# Climate-sensitivity competition

Predict `is_climate_sensitive`. Competition score = 0.60 x F1 + 0.40 x ROC-AUC.
Submission columns: `ID`, `TargetF1` (0/1), `TargetRAUC` (probability).

## Structure

```
src/                    shared, reusable code (imported by every notebook)
  config.py             paths, constants, column definitions
  features.py           feature engineering (same function for train/valid/test)
  metrics.py            competition score and threshold search
  validation.py         fold strategies and out-of-fold cross-validation
  models.py             model factory (logreg, RF, ExtraTrees, LightGBM, XGBoost, CatBoost)
  experiments.py        experiment tracker (EXP-001, EXP-002, ...)
  submission.py         submission builder with sanity checks
notebooks/              run in this order
  Data_audit.ipynb          raw files -> data/processed/*_merged.csv (needs data/raw/)
  Validation.ipynb          choice of validation strategy and threshold
  Baseline.ipynb            baseline scores by feature group
  Models_experiments.ipynb  model comparison, ensembles, submission
data/raw/               competition files (Train.csv, Test.csv, climate_features.csv, SampleSubmission.csv)
data/processed/         merged train/test files
outputs/                experiments_log.csv and submissions/
```

## Setup

```bash
pip install -r requirements.txt
jupyter lab
```

Open the notebooks from the `notebooks/` folder (they import `src` from the parent directory).
`Data_audit.ipynb` is only needed if you want to rebuild the merged files from `data/raw/`;
the merged files are already included.

## Key design decisions

- **Grouped validation by location.** About 97% of test rows come from locations that never
  appear in train, so validation folds must contain unseen locations only.
- **No `%run` between notebooks.** Shared logic lives in `src/`, so no notebook silently
  overwrites another one's variables.
- **`ID` and `deathdate_climate` are dropped.** The first is unique per row; the second is an
  exact copy of `deathdate` (a string that breaks CatBoost).
- **No early stopping on the validation fold**, and the threshold is chosen on out-of-fold
  predictions, to avoid optimistic scores.
- **Test predictions = average of the fold models**, consistent with the out-of-fold threshold.
