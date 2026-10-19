"""Central configuration: paths, constants and column definitions."""
from pathlib import Path

# Paths (resolved from this file, so they work from any working directory)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"
EXPERIMENT_LOG = OUTPUT_DIR / "experiments_log.csv"

TRAIN_MERGED = DATA_PROCESSED / "train_merged.csv"
TEST_MERGED = DATA_PROCESSED / "test_merged.csv"

# Reproducibility
RANDOM_STATE = 42
N_FOLDS = 5

# Columns
TARGET = "is_climate_sensitive"
ID_COL = "ID"
DATE_COL = "deathdate"
LOCATION_COL = "location"

# Exact duplicate of `deathdate` created by the climate merge, plus the ID.
# Neither carries predictive information and both break tree models
# (a date string is not a number, an ID is unique per row).
DROP_COLUMNS = [TARGET, ID_COL, "deathdate_climate"]

CATEGORICAL_COLUMNS = ["zone", "gender", LOCATION_COL]

# Variables that identify a place rather than describe an event.
LOCATION_FEATURES = [LOCATION_COL, "latitude", "longitude", "elevation", "slope"]

# Feature groups used in the baseline analysis
DEMOGRAPHIC_FEATURES = ["zone", "gender", "age"]
CALENDAR_FEATURES = ["death_year", "death_month", "death_dayofyear"]

# Competition metric: 0.60 * F1 + 0.40 * ROC-AUC
F1_WEIGHT = 0.60
AUC_WEIGHT = 0.40

# Submission format
SUBMISSION_COLUMNS = {"id": "ID", "label": "TargetF1", "proba": "TargetRAUC"}
