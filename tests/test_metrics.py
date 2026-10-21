"""Metric arithmetic must match the confusion matrix exactly (positive class = disease)."""
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from src.evaluation import METRICS, compute_metrics, get_score


def _hand_metrics(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(
        accuracy=(tp + tn) / (tp + tn + fp + fn),
        precision=tp / (tp + fp) if (tp + fp) else 0.0,
        recall=tp / (tp + fn) if (tp + fn) else 0.0,
        f1=(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0,
        tp=tp, fn=fn, fp=fp, tn=tn,
    )


def test_compute_metrics_matches_hand_arithmetic():
    y = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
    pred = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 1])
    got = compute_metrics(y, pred, score=y.astype(float))
    want = _hand_metrics(y, pred)
    for k in ["accuracy", "precision", "recall", "f1", "tp", "fn", "fp", "tn"]:
        assert abs(got[k] - want[k]) < 1e-12, f"{k}: {got[k]} vs {want[k]}"


def test_f1_lies_between_precision_and_recall():
    rng = np.random.default_rng(0)
    for _ in range(50):
        y = rng.integers(0, 2, 40)
        pred = rng.integers(0, 2, 40)
        m = compute_metrics(y, pred, score=y.astype(float))
        lo, hi = min(m["precision"], m["recall"]), max(m["precision"], m["recall"])
        assert lo - 1e-12 <= m["f1"] <= hi + 1e-12


def test_perfect_and_inverted_predictions():
    y = np.array([0, 0, 1, 1])
    perfect = compute_metrics(y, y, score=y.astype(float))
    assert perfect["accuracy"] == 1.0 and perfect["f1"] == 1.0 and perfect["auc"] == 1.0
    inverted = compute_metrics(y, 1 - y, score=(1 - y).astype(float))
    assert inverted["accuracy"] == 0.0 and inverted["f1"] == 0.0


def test_zero_division_is_safe():
    """A model that never predicts the positive class must not crash (zero_division=0)."""
    y = np.array([0, 0, 1, 1])
    pred = np.zeros(4, dtype=int)
    m = compute_metrics(y, pred, score=y.astype(float))
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0


def test_auc_is_threshold_free():
    """AUC depends on ranking, not the 0.5 threshold: rescaling the score keeps AUC."""
    y = np.array([0, 0, 0, 1, 1, 1])
    score = np.array([0.1, 0.2, 0.4, 0.35, 0.7, 0.9])
    a = roc_auc_score(y, score)
    b = roc_auc_score(y, score * 10 - 3)     # monotone transform
    assert abs(a - b) < 1e-12


def test_get_score_prefers_decision_function_then_proba():
    class FakeDF:
        def decision_function(self, X):
            return np.array([0.5])

    class FakeProba:
        def predict_proba(self, X):
            return np.array([[0.2, 0.8]])

    assert get_score(FakeDF(), None)[0] == 0.5
    assert get_score(FakeProba(), None)[0] == 0.8


def test_metric_names_constant():
    assert METRICS == ["accuracy", "precision", "recall", "f1", "auc"]
