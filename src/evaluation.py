"""Evaluation protocols A (60/40), B (70/30), C (10-fold CV) and isolation-forest outlier removal."""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split

from .data import RANDOM_STATE
from .models import make_preprocessor

PROTOCOLS = {"A": "60/40 split", "B": "70/30 split", "C": "10-fold CV"}
METRICS = ["accuracy", "precision", "recall", "f1", "auc"]


def get_score(model, X):
    """Continuous score used for ROC-AUC (SVM has no predict_proba unless probability=True)."""
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def compute_metrics(y, pred, score):
    """Positive class = heart disease present (1)."""
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(accuracy=accuracy_score(y, pred),
                precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0),
                f1=f1_score(y, pred, zero_division=0),
                auc=roc_auc_score(y, score), tp=tp, fn=fn, fp=fp, tn=tn)


def outlier_mask(X, contamination):
    """True = keep. IsolationForest on the imputed + scaled + one-hot features.
    contamination=None -> keep everything."""
    if contamination is None:
        return np.ones(len(X), dtype=bool)
    Z = make_preprocessor("linear").fit_transform(X)
    Z = Z.toarray() if sparse.issparse(Z) else Z
    iso = IsolationForest(contamination=contamination, random_state=RANDOM_STATE)
    return iso.fit_predict(Z) == 1


def _fit_eval(pipe, Xtr, ytr, Xte, yte, contamination):
    """Fit on (outlier-filtered) train, score both train and test. Outliers are only ever removed
    from TRAIN: the test set always contains every patient."""
    keep = outlier_mask(Xtr, contamination)
    model = clone(pipe).fit(Xtr[keep], ytr[keep])
    out = {"train": compute_metrics(ytr, model.predict(Xtr), get_score(model, Xtr)),
           "test": compute_metrics(yte, model.predict(Xte), get_score(model, Xte))}
    out["train"]["n_rows"] = int(keep.sum())
    out["test"]["n_rows"] = len(yte)
    return model, out


def run_protocol(pipe, X, y, protocol, contamination=None):
    """Returns (DataFrame with a 'train' and a 'test' row, extras).
    A/B: extras holds the fitted model and test data (for ROC curves).
    C  : metrics are mean over folds (+ *_std); confusion counts are summed over folds."""
    if protocol in ("A", "B"):
        test_size = {"A": 0.4, "B": 0.3}[protocol]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y,
                                              random_state=RANDOM_STATE)
        model, res = _fit_eval(pipe, Xtr, ytr, Xte, yte, contamination)
        rows = [dict(split=s, **res[s]) for s in ("train", "test")]
        extras = dict(model=model, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte,
                      score_test=get_score(model, Xte))
        return pd.DataFrame(rows), extras

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    folds = {"train": [], "test": []}
    for tr, te in skf.split(X, y):
        _, res = _fit_eval(pipe, X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te], contamination)
        for s in folds:
            folds[s].append(res[s])
    rows = []
    for s, lst in folds.items():
        f = pd.DataFrame(lst)
        row = dict(split=s)
        for m in METRICS:
            row[m], row[m + "_std"] = f[m].mean(), f[m].std(ddof=1)
        for c in ("tp", "fn", "fp", "tn", "n_rows"):
            row[c] = int(f[c].sum()) if c != "n_rows" else float(f[c].mean())
        rows.append(row)
    return pd.DataFrame(rows), None
