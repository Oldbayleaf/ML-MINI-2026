"""The pipeline must not leak: scaler/imputer statistics come from the training split only.

This is the single most important correctness property of the project. If the scaler were fit
on all 303 patients, test patients would influence training and every score would be inflated.
These tests turn that guarantee into something CI can enforce.
"""
import numpy as np
from sklearn.model_selection import train_test_split

from src.data import RANDOM_STATE, load_cleveland, split_xy
from src.models import make_preprocessor


def _split():
    X, y = split_xy(load_cleveland())
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)


def _dense(Z):
    return Z.toarray() if hasattr(Z, "toarray") else Z


def test_scaler_fit_on_train_only():
    """Scaled training columns have mean ~0; scaled test columns do not."""
    Xtr, Xte, _, _ = _split()
    prep = make_preprocessor("linear").fit(Xtr)          # fit on TRAIN only
    Ztr = _dense(prep.transform(Xtr))[:, :6]             # first 6 = numeric, scaled
    Zte = _dense(prep.transform(Xte))[:, :6]
    assert np.allclose(Ztr.mean(axis=0), 0, atol=1e-6), "train columns should be centred"
    assert not np.allclose(Zte.mean(axis=0), 0, atol=1e-3), (
        "test columns should NOT be centred (they did not influence the scaler)"
    )


def test_imputer_statistics_come_from_train():
    """Learned medians/modes match the training split, not the whole dataset."""
    Xtr, Xte, _, _ = _split()
    prep = make_preprocessor("linear").fit(Xtr)
    medians = dict(zip(
        ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"],
        prep.named_transformers_["num"]["imp"].statistics_,
    ))
    for col, stat in medians.items():
        assert stat == Xtr[col].median()


def test_preprocessor_handles_missing_and_unseen_categories():
    """A test row with a category never seen in training must not crash (handle_unknown)."""
    Xtr, _, _, _ = _split()
    prep = make_preprocessor("linear").fit(Xtr)
    weird = Xtr.iloc[[0]].copy()
    weird.loc[:, "cp"] = 99        # unseen code
    weird.loc[:, "age"] = np.nan   # missing numeric
    out = _dense(prep.transform(weird))
    assert out.shape[1] == prep.transform(Xtr).shape[1]
    assert np.isfinite(out).all()


def test_tree_preprocessor_keeps_integer_codes():
    """Tree/Naive-Bayes path imputes but does NOT scale or one-hot."""
    Xtr, _, _, _ = _split()
    prep = make_preprocessor("tree").fit(Xtr)
    assert _dense(prep.transform(Xtr)).shape[1] == 13    # still 13 columns
