"""Feature engineering. The same function is used for train, validation and test."""
import numpy as np
import pandas as pd

from .config import (
    CALENDAR_FEATURES,
    CATEGORICAL_COLUMNS,
    DATE_COL,
    DEMOGRAPHIC_FEATURES,
    DROP_COLUMNS,
    LOCATION_FEATURES,
)

FEATURE_SETS = ("base", "engineered")


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw date into numeric calendar features and drop the date."""
    df = df.copy()
    dates = pd.to_datetime(df.pop(DATE_COL), errors="coerce")
    df["death_year"] = dates.dt.year
    df["death_month"] = dates.dt.month
    df["death_dayofyear"] = dates.dt.dayofyear
    return df


def add_climate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ratios, differences and interactions built from the climate columns."""
    df = df.copy()

    def has(*cols):
        return set(cols).issubset(df.columns)

    # Temperature ranges
    if has("max_temperature", "min_temperature"):
        df["temp_range"] = df["max_temperature"] - df["min_temperature"]
    if has("tmax_30d", "tmin_30d"):
        df["temp_range_30d"] = df["tmax_30d"] - df["tmin_30d"]

    # Rainfall intensity and recent-versus-long-term rainfall
    if has("rain_sum_30d", "rain_days_30d"):
        df["rain_per_rainy_day_30d"] = df["rain_sum_30d"] / (df["rain_days_30d"] + 1)
    if has("rain_sum_7d", "rain_sum_30d"):
        df["rain_7d_ratio_30d"] = df["rain_sum_7d"] / (df["rain_sum_30d"] + 1e-6)
    if has("rain_sum_30d", "rain_sum_90d"):
        df["rain_30d_ratio_90d"] = df["rain_sum_30d"] / (df["rain_sum_90d"] + 1e-6)

    # Temperature anomalies
    if has("tavg_7d", "tavg_30d"):
        df["temp_recent_change"] = df["tavg_7d"] - df["tavg_30d"]
    if has("tavg_30d", "tavg_90d"):
        df["temp_medium_change"] = df["tavg_30d"] - df["tavg_90d"]

    # Vegetation change
    if has("ndvi_30d", "ndvi_90d"):
        df["ndvi_change"] = df["ndvi_30d"] - df["ndvi_90d"]

    # Heat / rainfall interaction
    if has("hot_days_30d", "rain_sum_30d"):
        df["heat_rain_interaction"] = df["hot_days_30d"] * df["rain_sum_30d"]

    return df


def prepare_features(
    df: pd.DataFrame,
    feature_set: str = "base",
    drop_location_features: bool = True,
) -> pd.DataFrame:
    """Build the model matrix from a merged dataframe.

    Parameters
    ----------
    df : merged train or test dataframe (the target column is ignored if present).
    feature_set : "base" (original variables + calendar) or
        "engineered" (base + derived climate features).
    drop_location_features : drop location, latitude, longitude, elevation and
        slope. Almost all test rows come from locations never seen in train,
        so these variables mostly help the model memorise places.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}, got {feature_set!r}")

    X = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns]).copy()
    X = add_calendar_features(X)

    if feature_set == "engineered":
        X = add_climate_features(X)

    if drop_location_features:
        X = X.drop(columns=[c for c in LOCATION_FEATURES if c in X.columns])

    # Force plain string categories (robust to pandas string dtypes)
    for col in CATEGORICAL_COLUMNS:
        if col in X.columns:
            X[col] = X[col].astype(str)

    return X.replace([np.inf, -np.inf], np.nan)


def get_categorical_features(X: pd.DataFrame) -> list:
    """Return every non-numeric column (works with object, string and category dtypes)."""
    return [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]


def get_feature_groups(X: pd.DataFrame) -> dict:
    """Split columns into demographic, calendar and climate groups."""
    demographic = [c for c in DEMOGRAPHIC_FEATURES if c in X.columns]
    calendar = [c for c in CALENDAR_FEATURES if c in X.columns]
    location = [c for c in LOCATION_FEATURES if c in X.columns]
    used = set(demographic + calendar + location)
    climate = [c for c in X.columns if c not in used]
    return {
        "demographic": demographic,
        "calendar": calendar,
        "location": location,
        "climate": climate,
    }
