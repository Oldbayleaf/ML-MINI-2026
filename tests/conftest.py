"""Shared fixtures for the replication test-suite.

The tests here are deliberately fast: they use a single 70/30 split (protocol B) rather than
the full 10-fold CV, so the whole suite runs in a few seconds. They guard the properties the
project's claims depend on — the data is the real Cleveland set, the pipeline leaks nothing,
and the metric arithmetic matches the confusion matrix.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_cleveland, split_xy  # noqa: E402
from src.models import MAIN_MODELS, build_pipeline  # noqa: E402


@pytest.fixture(scope="session")
def df():
    return load_cleveland()


@pytest.fixture(scope="session")
def xy(df):
    return split_xy(df)


@pytest.fixture(scope="session")
def fitted_b():
    """Every main model fitted once on the 70/30 training split (protocol B)."""
    from sklearn.model_selection import train_test_split

    from src.data import RANDOM_STATE

    df = load_cleveland()
    X, y = split_xy(df)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )
    fitted = {name: build_pipeline(name).fit(Xtr, ytr) for name in MAIN_MODELS}
    return {"models": fitted, "Xtr": Xtr, "Xte": Xte, "ytr": ytr, "yte": yte}
