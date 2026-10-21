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


# Expected shape of the UCI Cleveland data (the replication's whole premise rests on it).
N_ROWS = 303
N_MISSING = 6          # 4 in 'ca', 2 in 'thal'
CLASS_COUNTS = {0: 164, 1: 139}   # 164 healthy, 139 diseased (num > 0)


def load_cleveland(path=None, verify=True):
    """Read processed.cleveland.data. '?' becomes NaN. `target` = (num > 0).

    When `verify` is True (the default) the loaded data is checked against the known shape of
    the dataset, so a truncated or silently altered download fails loudly instead of quietly
    changing every downstream number.
    """
    path = path or DATA_DIR / "processed.cleveland.data"
    df = pd.read_csv(path, header=None, names=FEATURES + ["num"], na_values="?")
    df["target"] = (df["num"] > 0).astype(int)
    if verify:
        assert len(df) == N_ROWS, f"expected {N_ROWS} rows, got {len(df)}"
        assert int(df.isna().sum().sum()) == N_MISSING, "unexpected number of missing values"
        assert df["num"].between(0, 4).all(), "'num' must be an ordinal 0-4 severity"
        counts = df["target"].value_counts().to_dict()
        assert counts == CLASS_COUNTS, f"unexpected class balance: {counts}"
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
