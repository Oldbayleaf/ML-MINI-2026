"""The loaded data must be the real Cleveland set, and the integrity guards must bite."""
import pandas as pd
import pytest

from src.data import (BINARY, CATEGORICAL, CLASS_COUNTS, FEATURES, N_MISSING, N_ROWS,
                      NUMERIC, load_cleveland, split_xy)


def test_shape_and_columns(df):
    assert df.shape == (N_ROWS, 15)          # 13 features + num + target
    assert list(df.columns) == FEATURES + ["num", "target"]


def test_class_balance(df):
    assert df["target"].value_counts().to_dict() == CLASS_COUNTS


def test_missing_values(df):
    miss = df.isna().sum()
    assert int(miss.sum()) == N_MISSING
    assert int(miss["ca"]) == 4 and int(miss["thal"]) == 2


def test_num_is_ordinal_0_to_4(df):
    assert df["num"].between(0, 4).all()
    # target is exactly num > 0
    assert (df["target"] == (df["num"] > 0).astype(int)).all()


def test_feature_groups_partition_the_13_features():
    assert sorted(NUMERIC + BINARY + CATEGORICAL) == sorted(FEATURES)
    assert len(set(NUMERIC + BINARY + CATEGORICAL)) == len(FEATURES)   # no overlap


def test_split_xy_shapes(df):
    X, y = split_xy(df)
    assert X.shape == (N_ROWS, 13)
    assert y.shape == (N_ROWS,)
    assert set(y.unique()) == {0, 1}


def test_integrity_guard_catches_truncated_data(tmp_path):
    """A short/corrupt file must raise, not silently load."""
    bad = tmp_path / "bad.data"
    bad.write_text("63.0,1.0,1.0,145.0\n67.0,1.0,4.0,160.0\n")
    with pytest.raises(AssertionError):
        load_cleveland(bad)


def test_integrity_guard_can_be_disabled_for_raw_files(tmp_path):
    """verify=False still loads (used by tooling that reads arbitrary CSVs)."""
    src = load_cleveland()
    p = tmp_path / "copy.csv"
    src.drop(columns=["target"]).to_csv(p, header=False, index=False)
    out = load_cleveland(p, verify=False)
    assert len(out) == N_ROWS
