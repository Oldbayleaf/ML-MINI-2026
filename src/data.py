"""Data loading for the UCI Cleveland heart-disease replication."""
from pathlib import Path

import pandas as pd

RANDOM_STATE = 42

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

# Feature groups decide how each column is preprocessed (see models.make_preprocessor).
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]  # ca = 0..3 vessel count, kept numeric
BINARY = ["sex", "fbs", "exang"]                                    # already 0/1
CATEGORICAL = ["cp", "restecg", "slope", "thal"]                    # nominal codes -> one-hot for linear/KNN models

# The 7 features the paper reports (its Table 7).
PAPER_7 = ["age", "sex", "cp", "fbs", "chol", "slope", "ca"]


def load_cleveland(path=None):
    """Read processed.cleveland.data. '?' becomes NaN. `target` = (num > 0)."""
    path = path or DATA_DIR / "processed.cleveland.data"
    df = pd.read_csv(path, header=None, names=FEATURES + ["num"], na_values="?")
    df["target"] = (df["num"] > 0).astype(int)
    return df


def split_xy(df):
    return df[FEATURES].copy(), df["target"].copy()


def impute_whole_dataset(df):
    """Median/mode imputation on the WHOLE dataset.

    Used ONLY for descriptive EDA plots and the 'cleaned' CSV. Models never use this:
    they impute inside a Pipeline, fitted on the training split only.
    """
    out = df.copy()
    for c in NUMERIC:
        out[c] = out[c].fillna(out[c].median())
    for c in BINARY + CATEGORICAL:
        out[c] = out[c].fillna(out[c].mode()[0])
    return out
